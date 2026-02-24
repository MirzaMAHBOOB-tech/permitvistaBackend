#!/usr/bin/env python3
"""
tampa_pdfs_runner.py (updated)
- Adds USE_WEASYPRINT env var (auto/true/false)
- Adds compact logging to avoid giant console dumps
- Exposes logo_base64 and logo_image_url to templates
- Writes debug_render.html optionally (WRITE_DEBUG_HTML)
"""

import os
import json
import random
import base64
import tempfile
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
from dotenv import load_dotenv
from permit_portals import get_permit_portal_url
import pandas as pd
from slugify import slugify
from jinja2 import Template

# ------------------ CONFIG & ENV ------------------
load_dotenv()

# Existing config (you can still override with .env)
AZURE_CONN = os.getenv("AZURE_CONN") or None
SRC_CONTAINER = os.getenv("SRC_CONTAINER", "tampa-building-permits")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER", f"{SRC_CONTAINER}-pdfs")
LIMIT_PER_FILE = int(os.getenv("LIMIT_PER_FILE", "1"))
MAX_TOTAL_PDFS = int(os.getenv("MAX_TOTAL_PDFS", "0") or 0)
START_YEAR = int(os.getenv("START_YEAR", "2023"))
END_YEAR = int(os.getenv("END_YEAR", "2023"))
LOCAL_OUT_ROOT = os.getenv("LOCAL_OUT_ROOT", "enhanced_certificates_debug")
MAX_CSV_FILES = int(os.getenv("MAX_CSV_FILES", "5"))
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", None)

# New control flags
USE_WEASYPRINT = os.getenv("USE_WEASYPRINT", "auto").lower()  # 'auto'|'true'|'false'
VERBOSE = os.getenv("VERBOSE", "true").lower() in ("1","true","yes","y")
MAX_LIST_DISPLAY = int(os.getenv("MAX_LIST_DISPLAY", "6"))
WRITE_DEBUG_HTML = os.getenv("WRITE_DEBUG_HTML", "true").lower() in ("1","true","yes","y")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "templates", "certificate-placeholder.html")
LOCAL_SAMPLE_CSV = os.getenv("LOCAL_SAMPLE_CSV", r"D:\Tasks\permit-certs-project\data\tampa_permits.csv")

ID_CANDIDATES = ["_id", "PermitNumber", "PermitNum", "ID", "OBJECTID", "FID"]

# ------------------ Compact logger (ms precision) ------------------
def _now_ms():
    return int(time.perf_counter() * 1000)

START_T0 = _now_ms()

def log(msg: str, level: str = "info"):
    """Compact one-line logger. Respects VERBOSE for debug messages."""
    ms = _now_ms() - START_T0
    prefix = f"[{level.upper():5}] +{ms}ms"
    if level.lower() == "debug" and not VERBOSE:
        return
    print(f"{prefix} {msg}")

# ------------------ WeasyPrint detection w/ env control ------------------
HAVE_WEASYPRINT = False
_weasy_try_imported = False
if USE_WEASYPRINT == "false":
    log("USE_WEASYPRINT=false -> forcing fallback (pdfkit/wkhtmltopdf).", "info")
    HAVE_WEASYPRINT = False
elif USE_WEASYPRINT == "true":
    # Force import attempt and raise helpful error if missing
    try:
        from weasyprint import HTML  # type: ignore
        HAVE_WEASYPRINT = True
        _weasy_try_imported = True
        log("WeasyPrint successfully imported (forced).", "info")
    except Exception as e:
        log("USE_WEASYPRINT=true but WeasyPrint import failed: " + str(e), "error")
        raise RuntimeError("WeasyPrint requested via USE_WEASYPRINT=true but import failed. See WeasyPrint installation docs.") from e
else:
    # 'auto' behavior: try import but silently fall back
    try:
        from weasyprint import HTML  # type: ignore
        HAVE_WEASYPRINT = True
        _weasy_try_imported = True
        log("WeasyPrint available (auto mode).", "info")
    except Exception:
        HAVE_WEASYPRINT = False
        log("WeasyPrint not available (auto mode) — will use pdfkit/wkhtmltopdf fallback.", "info")

# ------------------ Azure SDK lazy import ------------------
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
except Exception:
    BlobServiceClient = None
    ContainerClient = None

# ------------------ GLOBALS ------------------
PROCESSED_FILES = 0
PROCESSED_PERMITS = 0

blob_service = None
src_container_client = None
out_container_client = None
if AZURE_CONN:
    if BlobServiceClient is None:
        log("azure-storage-blob package not installed; Azure mode disabled.", "warning")
        AZURE_CONN = None
    else:
        blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
        src_container_client = blob_service.get_container_client(SRC_CONTAINER)
        out_container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
        try:
            out_container_client.create_container()
        except Exception:
            pass
        log(f"Azure clients ready. Source: {SRC_CONTAINER} Output: {OUTPUT_CONTAINER}", "info")
else:
    log("AZURE_CONN not set; running local-only/test mode (will use local CSV).", "info")

# ------------------ Helpers ------------------
def detect_id_column(columns):
    for c in ID_CANDIDATES:
        if c in columns:
            return c
    return list(columns)[0] if columns else "certificate_number"

def format_timestamp(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip()
    if not s or s.lower() in {"0", "00000000", "nan", "nat", "null", "none"}:
        return ""
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.split("T")[0])
            return dt.strftime("%B %d, %Y")
        if "-" in s and len(s) >= 10:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        if len(s) == 13 and s.isdigit():
            dt = datetime.fromtimestamp(int(s) / 1000.0)
            return dt.strftime("%B %d, %Y")
        if len(s) == 10 and s.isdigit():
            dt = datetime.fromtimestamp(int(s))
            return dt.strftime("%B %d, %Y")
        if len(s) == 8 and s.isdigit():
            dt = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
            return dt.strftime("%B %d, %Y")
        if "/" in s:
            return s
        return s
    except Exception:
        return s

def generate_map_placeholder_url():
    return os.path.join(SCRIPT_DIR, "Medias", "map.png")

def _embed_image_as_data_uri(path, default_url=None):
    if not path:
        return default_url
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b = f.read()
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            data_uri = f"data:{mime};base64," + base64.b64encode(b).decode("ascii")
            return data_uri
    except Exception:
        pass
    return default_url

def _read_image_base64(path):
    """Return pure base64 string (no data: prefix) or None."""
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        pass
    return None

def upload_pdf_bytes_to_azure(container_client: ContainerClient, blob_path: str, pdf_bytes: bytes):
    try:
        container_client.upload_blob(blob_path, pdf_bytes, overwrite=True)
        return True
    except Exception as e:
        log("Azure upload failed: " + str(e), "error")
        return False


# ------------------ Rendering ------------------
def render_enhanced_certificate_pdf_bytes(record: dict, id_col: str, all_fields: list[str]) -> bytes:
    if not os.path.exists(HTML_TEMPLATE_PATH):
        raise FileNotFoundError(f"HTML template not found: {HTML_TEMPLATE_PATH}")

    with open(HTML_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        tpl = Template(f.read())

    # logo candidates (same as before)
    logo_candidates = [
        os.path.join(SCRIPT_DIR, "Medias", "badge.png"),
        os.path.join(SCRIPT_DIR, "Medias", "badge.jpg"),
        os.path.join(SCRIPT_DIR, "Medias", "badge.jpeg"),
    ]
    logo_data_uri = None
    logo_base64 = None
    for p in logo_candidates:
        if os.path.exists(p):
            logo_data_uri = _embed_image_as_data_uri(p)
            logo_base64 = _read_image_base64(p)
            break

    if not logo_data_uri:
        logo_data_uri = "https://placehold.co/80x80/ffd700/8b4513?text=PERMIT+LOGO+ERROR"
    if not logo_base64:
        # attempt to extract base64 from data URI if available
        if logo_data_uri.startswith("data:"):
            logo_base64 = logo_data_uri.split(",")[1]
        else:
            logo_base64 = ""

    ctx = {
        "certificate_number": generate_certificate_number(),
        "generated_on_date": datetime.now().strftime("%m/%d/%Y"),

        "property": {
            "address_description": record.get("AddressDescription", "") or record.get("AddressDescription1", "") or record.get("ParcelID", ""),
            "address": record.get("Address", "") or record.get("StreetAddress", "") or record.get("PropertyAddress", "") or record.get("OriginalAddress1", ""),
            "city": record.get("City", "") or record.get("OriginalCity", "") or record.get("PropertyCity", ""),
            "state": record.get("State", "") or record.get("OriginalState", "") or record.get("PropertyState", ""),
            "zip_code": record.get("Zip", "") or record.get("OriginalZip", "") or record.get("ZIP", ""),
            "pin": record.get("PIN", "") or record.get("ParcelID", "") or record.get("PropertyID", ""),
        },
        "permit": {
            "class": record.get("PermitClass", "") or record.get("Class", ""),
            "classification": record.get("PermitClassMapped", "") or record.get("PermitClassification", ""),
            "number": record.get("PermitNumber", "") or record.get("PermitNum", "") or record.get("_id", "") or record.get("ApplicationNumber", ""),
            "type": record.get("PermitType", "") or record.get("Type", ""),
            "type_classification": record.get("PermitTypeMapped", "") or record.get("TypeClassification", ""),
            "id": record.get(id_col, ""),
        },
        "dates": {
            "application": format_timestamp(record.get("AppliedDate", "") or record.get("ApplicationDate", "")),
            "completion": format_timestamp(record.get("CompletedDate", "") or record.get("CompletionDate", "")),
            "expires": format_timestamp(record.get("ExpirationDate", "") or record.get("ExpiresDate", "")),
            "last_updated": format_timestamp(record.get("LastUpdated", "") or record.get("UpdatedDate", "")),
            "status_date": format_timestamp(record.get("StatusDate", "") or record.get("StatusDate", "")),
        },
        "status_current_color": "#DC2626",
        "status_current": record.get("StatusCurrentMapped", "") or record.get("CurrentStatus", "") or record.get("PermitStatus", ""),
        "current_status": record.get("CurrentStatus", "") or record.get("StatusCurrentMapped", "") or record.get("PermitStatus", ""),
        "other": {
            "online_record_url": (
                record.get("Link", "") or record.get("OnlineRecord", "")
                or get_permit_portal_url(
                    record.get("Jurisdiction", "") or record.get("OriginalCity", "") or "TAMPA"
                )
                or ""
            ),
            "publisher": record.get("Publisher", "PermitVista"),
        },
        "work_description": record.get("WorkDescription", "") or record.get("ProjectDescription", "") or record.get("Description", ""),
        "map_image_url": generate_map_placeholder_url(),
        # both forms provided for templates that expect data URI or plain base64
        "logo_image_url": logo_data_uri,
        "logo_base64": logo_base64,
    }

    filled_html = tpl.render(ctx)

    # always write debug html when enabled so you can inspect if needed
    if WRITE_DEBUG_HTML:
        try:
            debug_html_path = os.path.join(SCRIPT_DIR, "debug_render.html")
            with open(debug_html_path, "w", encoding="utf-8") as fh:
                fh.write(filled_html)
            log(f"Wrote debug HTML: {debug_html_path}", "info")
        except Exception as e:
            log("Failed to write debug HTML: " + str(e), "warning")

    base_dir = SCRIPT_DIR

    # Use WeasyPrint when enabled
    if HAVE_WEASYPRINT:
        try:
            from weasyprint import HTML as _HTML
            t0 = _now_ms()
            pdf_bytes = _HTML(string=filled_html, base_url=base_dir).write_pdf()
            t1 = _now_ms()
            log(f"WeasyPrint rendered PDF in {t1 - t0}ms", "info")
            if isinstance(pdf_bytes, (bytes, bytearray)):
                return bytes(pdf_bytes)
        except Exception as e:
            log("WeasyPrint render failed, falling back to pdfkit/wkhtmltopdf: " + str(e), "warning")

    # Fallback to pdfkit/wkhtmltopdf
    try:
        import pdfkit
    except Exception as e:
        raise RuntimeError("PDF rendering failed: pdfkit not installed and WeasyPrint not available.") from e

    wk_path = WKHTMLTOPDF_PATH or shutil.which("wkhtmltopdf")
    config = pdfkit.configuration(wkhtmltopdf=wk_path) if wk_path else None
    options = {
        "enable-local-file-access": None,
        "page-size": "Letter",
        "margin-top": "0.3in",
        "margin-bottom": "0.5in",
        "margin-left": "0.3in",
        "margin-right": "0.3in",
    }
    t0 = _now_ms()
    pdf_bytes = pdfkit.from_string(filled_html, False, options=options, configuration=config)
    t1 = _now_ms()
    log(f"pdfkit rendered PDF in {t1 - t0}ms (wkhtmltopdf:{bool(wk_path)})", "info")
    return bytes(pdf_bytes)

# ------------------ Save locally ------------------
def save_pdf_local(pdf_bytes: bytes, year: str, month: str, day: str, file_name: str):
    out_dir = os.path.join(LOCAL_OUT_ROOT, year, month, day)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path

# ------------------ Blob scanning/processing ------------------
def iter_csv_blobs_by_year(start_year: int, end_year: int):
    if not src_container_client:
        return []

    found = []
    detected_years = set()
    try:
        for blob in src_container_client.list_blobs():
            name = blob.name.strip()
            if not name.lower().endswith(".csv"):
                continue
            parts = name.split("/", 1)
            if parts and parts[0].isdigit():
                detected_years.add(int(parts[0]))
    except Exception as e:
        log("Azure list_blobs initial scan failed: " + str(e), "warning")

    if detected_years:
        min_year, max_year = min(detected_years), max(detected_years)
        start_year = min(start_year, min_year)
        end_year = max(end_year, max(detected_years))
        log(f"Detected year folders in Azure: {sorted(detected_years)[:MAX_LIST_DISPLAY]} (total {len(detected_years)})", "info")
        log(f"Adjusted scan range: {start_year} → {end_year}", "info")

    # collect CSV names but don't dump entire list to console
    for year in range(end_year, start_year - 1, -1):
        prefix = f"{year}/"
        try:
            for b in src_container_client.list_blobs(name_starts_with=prefix):
                name = (b.name or "").strip()
                if name.lower().endswith(".csv") and not name.endswith("/"):
                    found.append(name)
        except Exception as e:
            log(f"Azure list_blobs failed for year {year}: {e}", "warning")
            continue
        if len(found) >= MAX_CSV_FILES:
            break

    if found:
        # print just count and first few items to avoid flooding console
        preview = found[:MAX_LIST_DISPLAY]
        log(f"Found {len(found)} CSV(s) (showing up to {MAX_LIST_DISPLAY}): {preview}", "info")
    else:
        log("No CSVs found in Azure container (or listing failed).", "info")

    return found[:MAX_CSV_FILES]

def extract_ymd(record: dict):
    for field in ["AppliedDate", "IssuedDate", "StatusDate", "CompletedDate", "COIssuedDate"]:
        if field in record and record[field]:
            s = str(record[field]).strip()
            try:
                if "T" in s:
                    dt = datetime.strptime(s.split("T")[0], "%Y-%m-%d")
                    return f"{dt.year:04d}", f"{dt.month:02d}", f"{dt.day:02d}"
                if "-" in s and len(s) >= 10:
                    dt = datetime.strptime(s[:10], "%Y-%m-%d")
                    return f"{dt.year:04d}", f"{dt.month:02d}", f"{dt.day:02d}"
                if len(s) == 8 and s.isdigit():
                    return s[:4], s[4:6], s[6:8]
            except Exception:
                pass
    now = datetime.now()
    return f"{now.year:04d}", f"{now.month:02d}", f"{now.day:02d}"

def process_one_blob(blob_name: str, process_start):
    global PROCESSED_PERMITS
    log(f"Processing CSV: {blob_name}", "info")
    try:
        blob_client = src_container_client.get_blob_client(blob_name)
        props = blob_client.get_blob_properties()
        if props.size == 0:
            log(f"Skipping empty blob: {blob_name}", "warning")
            return
        log(f"Downloading blob ({props.size} bytes): {blob_name}", "debug")
        data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(data), dtype=str)
    except Exception as e:
        log(f"Failed to download or parse CSV: {blob_name} | {e}", "error")
        return

    if df.empty:
        log(f"Empty CSV: {blob_name}", "warning")
        return

    id_col = detect_id_column(df.columns)
    all_fields = list(df.columns)
    orig = len(df)
    df = df.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])

    if LIMIT_PER_FILE and LIMIT_PER_FILE > 0:
        df = df.head(LIMIT_PER_FILE)
        log(f"LIMIT_PER_FILE active: only first {LIMIT_PER_FILE} rows will be processed from this CSV.", "info")

    log(f"Records: {orig} -> {len(df)} after filtering", "info")
    made = 0
    for idx, row in df.iterrows():
        if MAX_TOTAL_PDFS and PROCESSED_PERMITS >= MAX_TOTAL_PDFS:
            log(f"Reached MAX_TOTAL_PDFS ({MAX_TOTAL_PDFS}); stopping.", "info")
            break

        rec = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        y, m, d = extract_ymd(rec)
        rid = (rec.get(id_col) or "").strip() or f"row-{idx}"
        fname = f"enhanced_{slugify(rid)}_{random.randint(1000,9999)}.pdf"

        try:
            pdf_bytes = render_enhanced_certificate_pdf_bytes(rec, id_col, all_fields)

            if out_container_client:
                blob_path = f"{y}/{m}/{d}/{fname}"
                ok = upload_pdf_bytes_to_azure(out_container_client, blob_path, pdf_bytes)
                if ok:
                    log(f"Uploaded PDF to blob: {OUTPUT_CONTAINER}/{blob_path}", "info")
                else:
                    log("Upload to Azure returned False.", "warning")
            else:
                ok = False

            local_path = save_pdf_local(pdf_bytes, y, m, d, fname)
            log(f"Saved local PDF copy: {local_path}", "info")

            made += 1
            PROCESSED_PERMITS += 1
        except Exception as e:
            log(f"Failed to render/upload record {idx + 1}: {e}", "warning")

    elapsed_ms = _now_ms() - int(process_start * 1000) if isinstance(process_start, float) else 0
    log(f"Completed CSV: {blob_name} | Made: {made} | Elapsed: {elapsed_ms/1000:.2f} sec", "info")

def process_local_csv(local_csv_path: str):
    global PROCESSED_PERMITS
    log(f"Processing local CSV: {local_csv_path}", "info")
    if not os.path.exists(local_csv_path):
        log("Local CSV not found: " + local_csv_path, "error")
        return
    df = pd.read_csv(local_csv_path, dtype=str)
    if df.empty:
        log("Local CSV empty: " + local_csv_path, "warning")
        return

    id_col = detect_id_column(df.columns)
    all_fields = list(df.columns)
    orig = len(df)
    df = df.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])

    if LIMIT_PER_FILE and LIMIT_PER_FILE > 0:
        df = df.head(LIMIT_PER_FILE)
        log(f"LIMIT_PER_FILE active for local CSV: only first {LIMIT_PER_FILE} rows will be processed.", "info")

    log(f"Records: {orig} -> {len(df)} after filtering (local)", "info")
    made = 0
    process_start = time.perf_counter()
    for idx, row in df.iterrows():
        if MAX_TOTAL_PDFS and PROCESSED_PERMITS >= MAX_TOTAL_PDFS:
            log(f"Reached MAX_TOTAL_PDFS ({MAX_TOTAL_PDFS}); stopping.", "info")
            break
        rec = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        y, m, d = extract_ymd(rec)
        rid = (rec.get(id_col) or "").strip() or f"row-{idx}"
        fname = f"enhanced_local_{slugify(rid)}_{random.randint(1000,9999)}.pdf"

        try:
            pdf_bytes = render_enhanced_certificate_pdf_bytes(rec, id_col, all_fields)
            local_path = save_pdf_local(pdf_bytes, y, m, d, fname)
            log("Saved local PDF: " + local_path, "info")
            made += 1
            PROCESSED_PERMITS += 1
        except Exception as e:
            log("Failed to render local record: " + str(e), "warning")

    elapsed = time.perf_counter() - process_start
    log(f"Completed local CSV: {local_csv_path} | Made: {made} | Elapsed: {elapsed:.2f} sec", "info")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    start_time = time.perf_counter()
    log("Starting PDF generation: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "info")

    csv_list = []
    if AZURE_CONN and src_container_client:
        csv_list = iter_csv_blobs_by_year(START_YEAR, END_YEAR)
        if csv_list:
            log(f"Proceeding with {len(csv_list)} CSV(s) from Azure (limited to {MAX_CSV_FILES}).", "info")
        else:
            log("No CSVs found in Azure - will fall back to local CSV.", "info")
    else:
        log("Azure not used or unavailable; will use local fallback CSV.", "info")

    if not csv_list:
        process_local_csv(LOCAL_SAMPLE_CSV)
    else:
        for i, blob_name in enumerate(csv_list, start=1):
            if MAX_TOTAL_PDFS and PROCESSED_PERMITS >= MAX_TOTAL_PDFS:
                log("Global cap reached; exiting.", "info")
                break
            log(f"File {i}: {blob_name}", "info")
            process_one_blob(blob_name, start_time)
            PROCESSED_FILES += 1

    total_time = time.perf_counter() - start_time
    log("Run complete", "info")
    log(f"Files processed: {PROCESSED_FILES}", "info")
    log(f"PDFs generated/uploaded/saved: {PROCESSED_PERMITS}", "info")
    log(f"Total time: {total_time:.2f} seconds", "info")
