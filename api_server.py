# api_server.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO
import os
import tempfile
import logging
import subprocess
import time
import threading
import webbrowser
from typing import Optional, List, Tuple
import re

# ----------------- Setup / Logging -----------------
BASE_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# reduce azure noise (set to ERROR)
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.storage").setLevel(logging.ERROR)

# ----------------- Check WeasyPrint availability at startup -----------------
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
    logging.info("WeasyPrint is available and ready")
except ImportError as e:
    logging.warning("WeasyPrint import failed at startup: %s - PDF generation will use fallback methods", e)
except Exception as e:
    logging.warning("WeasyPrint check failed at startup: %s - PDF generation will use fallback methods", e)

# ----------------- Load environment -----------------
env_path = BASE_DIR / ".env"
load_dotenv(env_path)
AZURE_CONN = os.getenv("AZURE_CONN")
SRC_CONTAINER = os.getenv("SRC_CONTAINER")
OUTPUT_CONTAINER = os.getenv("OUTPUT_CONTAINER")
SAMPLE_PERMIT_LIMIT = int(os.getenv("SAMPLE_PERMIT_LIMIT", "5"))
MAX_CSV_FILES = int(os.getenv("MAX_CSV_FILES", "10"))

# ----------------- FastAPI app -----------------
app = FastAPI(title="Permit Certificates", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount static assets from filesystem using absolute paths
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/Medias", StaticFiles(directory=str(BASE_DIR / "Medias")), name="Medias")

TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ----------------- Azure Client -----------------
blob_service = None
src_container = None
pdf_container = None
if AZURE_CONN:
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
        if SRC_CONTAINER:
            src_container = blob_service.get_container_client(SRC_CONTAINER)
        if OUTPUT_CONTAINER:
            pdf_container = blob_service.get_container_client(OUTPUT_CONTAINER)
        logging.info("Azure clients initialized. src_container=%s output=%s", SRC_CONTAINER, OUTPUT_CONTAINER)
    except Exception as e:
        logging.exception("Azure init failed; continuing without Azure: %s", e)
        blob_service = None
        src_container = None
        pdf_container = None
else:
    logging.warning("AZURE_CONN not set; server will not find data")

ID_CANDIDATES = ["PermitNumber", "PermitNum", "_id", "ID", "OBJECTID", "FID", "ApplicationNumber"]

# ----------------- Helpers -----------------
def _read_csv_bytes_from_blob(container_client, blob_name: str) -> Optional[pd.DataFrame]:
    """Read whole CSV blob into a pandas DataFrame (strings)."""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(data), dtype=str)
        return df
    except Exception as e:
        logging.exception("Failed to read CSV blob %s: %s", blob_name, e)
        return None

def pick_id_from_record(record: dict) -> str:
    for candidate in ID_CANDIDATES:
        if candidate in record and record.get(candidate):
            return str(record[candidate])
    # fallback to any non-empty field
    for k, v in record.items():
        if v:
            return str(v)
    return "unknown"

def tmp_pdf_path_for_id(permit_id: str) -> str:
    """Temporary PDF path for a given permit id."""
    safe = permit_id.replace("/", "_").replace("\\", "_")
    return os.path.join(tempfile.gettempdir(), f"permit_{safe}.pdf")

def _find_record_in_azure_csvs(permit_id: str):
    """Search ONLY azure csv blobs for a specific permit id. Return one record dict or None."""
    if not src_container:
        logging.warning("_find_record_in_azure_csvs called but src_container is not configured")
        return None
    try:
        scanned = 0
        pid_lower = (permit_id or "").strip().lower()
        for blob in src_container.list_blobs():
            if scanned >= MAX_CSV_FILES:
                logging.info("Reached MAX_CSV_FILES scan limit (%d)", MAX_CSV_FILES)
                break
            name = getattr(blob, "name", str(blob))
            if not name.lower().endswith(".csv"):
                continue
            df = _read_csv_bytes_from_blob(src_container, name)
            scanned += 1
            if df is None:
                continue
            for _, row in df.iterrows():
                rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                pid = pick_id_from_record(rec)
                if pid and pid.strip().lower() == pid_lower:
                    logging.info("Matched permit %s in blob %s", permit_id, name)
                    return rec
    except Exception as e:
        logging.exception("Error searching Azure CSVs: %s", e)
    return None

# Address parsing helpers used for server-side matching
ZIP_RE = re.compile(r"(\d{5})(?:-\d{4})?$")
NUM_RE = re.compile(r"^\s*(\d+)\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    return NON_ALNUM.sub(" ", s.lower()).strip()

SUFFIX_MAP = {
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "road": "rd", "rd": "rd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct",
    "terrace": "ter", "ter": "ter",
    "place": "pl", "pl": "pl",
    "way": "wy", "wy": "wy",
    "circle": "cir", "cir": "cir",
    "trail": "trl", "trl": "trl",
    "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy",
    "driveway": "drwy", "drwy": "drwy",
    "highwy": "hwy",
}
DIR_MAP = {
    "north": "n", "n": "n",
    "south": "s", "s": "s",
    "east": "e", "e": "e",
    "west": "w", "w": "w",
    "ne": "ne", "northeast": "ne",
    "nw": "nw", "northwest": "nw",
    "se": "se", "southeast": "se",
    "sw": "sw", "southwest": "sw",
}

def normalize_suffix(token: str) -> str:
    t = (token or "").lower().strip(". ")
    return SUFFIX_MAP.get(t, t)

def normalize_dir(token: str) -> str:
    t = (token or "").lower().strip(". ")
    return DIR_MAP.get(t, t)

def extract_address_components(input_address: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Heuristic extract of (street_number, route_text, zip_code) from a provided address string.
    Handles unit markers (A/B, Apt, Unit, #), trailing directional tokens (N, S, E, W, NE, etc.)
    and common punctuation.
    Returns (street_number or None, route_text or None, zip or None)
    """
    if not input_address:
        return None, None, None
    s = input_address.strip()

    # zip
    zip_match = ZIP_RE.search(s)
    zip_code = zip_match.group(1) if zip_match else None

    # street number (leading number like "2509")
    num_match = NUM_RE.match(s)
    street_number = num_match.group(1) if num_match else None

    # remove leading number for route extraction
    route = s
    if street_number:
        route = route[len(street_number):].strip()

    # drop trailing comma-delimited city/state/zip portion: keep only part before first comma
    if "," in route:
        route = route.split(",")[0].strip()

    # remove unit tokens (A/B, UNIT, APT, STE, #, FL, LOT) that often appear after route
    route = re.sub(r"\b(?:a\/b|apt|unit|ste|suite|#|fl|floor|lot)\b[\w\s\-\/]*$", "", route, flags=re.IGNORECASE).strip()

    # remove trailing directional (e.g., "N", "W", "NE", "SW") that may follow street type
    route = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?$", "", route, flags=re.IGNORECASE).strip()

    # normalize whitespace and punctuation to simple tokens (lowercase)
    route_norm = normalize_text(route)

    return street_number, route_norm if route_norm else None, zip_code

def record_address_values(rec: dict) -> List[str]:
    """
    Collect address-like fields from a record and normalize them for match testing.
    Returns list of normalized address strings to test against.
    """
    candidates = []
    # Broadened set of field names commonly seen across datasets
    possible_fields = [
        # direct address lines
        "Address", "Address1", "Address2", "AddressLine1", "AddressLine2", "AddressLine",
        "OriginalAddress", "OriginalAddress1", "OriginalAddress2",
        "PropertyAddress", "StreetAddress", "FullAddress", "AddressFull",
        "SitusAddress", "SiteAddress", "Location", "Site Location", "AddressDescription",
        # components which we may later compose
        "Street", "StreetName", "Street1", "Street2",
    ]
    for k in possible_fields:
        v = rec.get(k)
        if v and isinstance(v, str) and v.strip():
            # remove common unit markers and ACA tag to avoid mismatch
            cleaned = re.sub(r"\b(?:a\/b|apt|unit|ste|suite|#|fl|floor|lot)\b[\w\s\-\/]*$", "", v, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"\s*\(aca\)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
            candidates.append(normalize_text(cleaned))

    # Common triplet composition attempts
    addr = rec.get("Address") or rec.get("OriginalAddress") or rec.get("StreetAddress") or rec.get("PropertyAddress") or ""
    city = rec.get("OriginalCity") or rec.get("City") or rec.get("PropertyCity") or ""
    zipc = rec.get("OriginalZip") or rec.get("ZipCode") or rec.get("ZIP") or rec.get("Zip") or ""
    if addr or city or zipc:
        combined = f"{addr} {city} {zipc}"
        combined = re.sub(r"\b(?:a\/b|apt|unit|ste|suite|#|fl|floor|lot)\b[\w\s\-\/]*$", "", combined, flags=re.IGNORECASE).strip()
        combined = re.sub(r"\s*\(aca\)\s*$", "", combined, flags=re.IGNORECASE).strip()
        candidates.append(normalize_text(combined))

    # Compose from granular street components when present
    number = rec.get("StreetNumber") or rec.get("AddrNum") or rec.get("HouseNumber") or ""
    name = rec.get("StreetName") or rec.get("Street") or ""
    stype = rec.get("StreetType") or rec.get("StreetSuffix") or ""
    sdir = rec.get("StreetDir") or rec.get("PreDir") or rec.get("PostDir") or ""
    composed_parts = " ".join([str(number or "").strip(), str(name or "").strip(), str(stype or "").strip(), str(sdir or "").strip()]).strip()
    if composed_parts:
        comp1 = composed_parts
        comp2 = " ".join([composed_parts, str(city or "").strip(), str(zipc or "").strip()]).strip()
        candidates.append(normalize_text(comp1))
        candidates.append(normalize_text(comp2))

    # remove duplicates while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def canonicalize_address_main(s: str) -> Tuple[str, List[str]]:
    """
    Produce a canonical key from a street-like string (before commas, no city/zip).
    Returns (street_number or "", tokens_without_number) where tokens include direction/suffix normalized.
    """
    if not s:
        return "", []
    # take before first comma
    main = s.split(",")[0]
    # strip '(ACA)'
    main = re.sub(r"\s*\(aca\)\s*$", "", main, flags=re.IGNORECASE).strip()
    norm = normalize_text(main)
    if not norm:
        return "", []
    tokens = [t for t in norm.split() if t]
    if not tokens:
        return "", []
    street_num = tokens[0] if tokens and tokens[0].isdigit() else ""
    rest = tokens[1:] if street_num else tokens[:]
    # normalize dir/suffix tokens
    rest_norm = []
    for t in rest:
        dt = normalize_dir(t)
        st = normalize_suffix(t)
        if dt in DIR_MAP.values():
            rest_norm.append(dt)
        elif st in SUFFIX_MAP.values():
            rest_norm.append(st)
        else:
            rest_norm.append(t)
    # move single-letter cardinal from end to be near start for better comparison
    if rest_norm and rest_norm[-1] in ("n", "s", "e", "w", "ne", "nw", "se", "sw"):
        d = rest_norm.pop()
        rest_norm.insert(0, d)
    return street_num, rest_norm

# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    return JSONResponse({
        "azure_connected": bool(src_container),
        "src_container": SRC_CONTAINER,
        "output_container": OUTPUT_CONTAINER
    })

@app.get("/facets")
def facets(field: str = Query(...), max_items: int = Query(200)):
    values: List[str] = []
    sample_records: List[dict] = []
    max_items = max(1, min(int(max_items or 200), 5000))
    is_permit_id_req = field.lower() in ("permit", "permit_number", "permitnumber", "permitnum", "ticket", "ticket_number", "ticketnumber")

    if not src_container:
        return JSONResponse({"field": field, "values": [], "samples": [], "error": "SRC_CONTAINER not configured or cannot connect to Azure."})

    try:
        scanned = 0
        for blob in src_container.list_blobs():
            if scanned >= MAX_CSV_FILES:
                break
            name = getattr(blob, "name", str(blob))
            if not name.lower().endswith(".csv"):
                continue
            df = _read_csv_bytes_from_blob(src_container, name)
            scanned += 1
            if df is None:
                continue
            cols = [c.strip() for c in df.columns.tolist()]

            if is_permit_id_req:
                for _, row in df.iterrows():
                    rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                    pid = pick_id_from_record(rec) or ""
                    if pid:
                        if pid not in values:
                            values.append(pid)
                        if len(sample_records) < SAMPLE_PERMIT_LIMIT:
                            sample_records.append(rec)
                        if len(sample_records) >= SAMPLE_PERMIT_LIMIT:
                            break
                if len(sample_records) >= SAMPLE_PERMIT_LIMIT:
                    break
            else:
                for c in cols:
                    if c.lower() == field.lower():
                        for v in df[c].dropna().astype(str):
                            s = v.strip()
                            if s and s not in values:
                                values.append(s)
                        break
            if not is_permit_id_req and len(values) >= max_items:
                break
    except Exception as e:
        logging.exception("FACETS scan error: %s", e)

    return JSONResponse({"field": field, "values": values[:max_items], "samples": sample_records})

def get_local_pdf_path_if_exists(permit_id: str) -> Optional[str]:
    """Return path if a cached/generated PDF exists for permit_id."""
    p = tmp_pdf_path_for_id(permit_id)
    return p if os.path.exists(p) else None

@app.get("/search")
def search(
    address: str = Query(..., min_length=3, description="Main search address — REQUIRED (from Google Autocomplete)"),
    city: Optional[str] = Query(None, description="Optional city"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    max_results: int = Query(200),
    scan_limit: Optional[int] = Query(None, description="Override number of CSV blobs to scan (bounds 1..5000)"),
    permit: Optional[str] = Query(None, description="Optional permit number to match after address"),
    street_number_q: Optional[str] = Query(None, description="Optional street number from Google selection"),
    street_name_q: Optional[str] = Query(None, description="Optional street name from Google selection"),
    street_type_q: Optional[str] = Query(None, description="Optional street type/suffix from Google selection"),
    street_dir_q: Optional[str] = Query(None, description="Optional street direction (N/E/S/W) from Google selection"),
    zip_q: Optional[str] = Query(None, description="Optional zip code from Google selection")
):
    """
    Search records by address primarily. 'address' is required.
    Matching strategy:
     - try to extract street number, route (normalized), and zip from the provided address
     - for each CSV record, normalize address-like fields and attempt to match:
         * if zip is present in input, require zip match (strong)
         * if street number present, prefer records whose normalized address starts with that number
         * route (street name) is tested as substring on normalized forms
     - optional 'city', 'date_from', 'date_to' are applied as filters
    """
    results: List[dict] = []
    if not src_container:
        return JSONResponse({"results": [], "error": "SRC_CONTAINER not configured"})

    # Normalize input and extract heuristics
    input_addr = (address or "").strip()
    if not input_addr:
        # Query(...) already enforces presence, but double-check
        raise HTTPException(status_code=400, detail="address parameter required")

    street_number, route_norm, zip_code = extract_address_components(input_addr)
    route_norm = route_norm or normalize_text(input_addr)
    city_norm = normalize_text(city or "")

    # Build canonical address key from provided structured parts (if any)
    google_main = input_addr.split(",")[0].strip()
    if street_number_q or street_name_q or street_type_q or street_dir_q:
        # Build "dir street_name street_type" with direction first, like AddressDescription style
        dir_tok = normalize_dir(street_dir_q or "")
        type_tok = normalize_suffix(street_type_q or "")
        name_tok = normalize_text(street_name_q or "")
        number_tok = (street_number_q or "").strip()
        rebuilt = " ".join(t for t in [number_tok, dir_tok, name_tok, type_tok] if t).strip()
        if rebuilt:
            google_main = rebuilt
    # Canonical key tuple for input
    in_num, in_tokens = canonicalize_address_main(google_main)
    zip_from_google = (zip_q or "").strip() or (extract_address_components(input_addr)[2] or "")

    # utility: normalize the main street portion (before the first comma)
    def main_street_part(s: str) -> str:
        if not s:
            return ""
        part = s.split(",")[0].strip()
        # remove unit/apt markers at end
        part = re.sub(r"\b(?:a\/b|apt|unit|ste|suite|#|fl|floor|lot)\b[\w\s\-\/]*$", "", part, flags=re.IGNORECASE).strip()
        # drop trailing one-letter cardinal directions (N,S,E,W)
        part = re.sub(r"\b(N|S|E|W)\b\.?$", "", part, flags=re.IGNORECASE).strip()
        return normalize_text(part)

    # Log incoming search intent
    logging.info("SEARCH start | address='%s' city='%s' permit='%s' dates=%s..%s max=%s scan_limit=%s | google_parts num=%s name_tokens=%s zip=%s",
                 input_addr, city, permit, date_from, date_to, max_results, scan_limit, in_num, in_tokens, zip_from_google)

    start_t = time.perf_counter()
    tries = 0
    seen_ids: set = set()
    canonical_hit = False

    try:
        scan_max = min(max(1, int(scan_limit or MAX_CSV_FILES)), 5000)
        scanned = 0
        for blob in src_container.list_blobs():
            if scanned >= scan_max:
                logging.info("Reached scan limit during search (%d)", scan_max)
                break
            name = getattr(blob, "name", str(blob))
            if not name.lower().endswith(".csv"):
                continue

            df = _read_csv_bytes_from_blob(src_container, name)
            scanned += 1
            if df is None:
                continue
            try:
                df_len = len(df)
            except Exception:
                df_len = -1
            if scanned % 10 == 1:
                logging.info("Scanning blob %d/%d: %s rows=%s", scanned, scan_max, name, df_len if df_len >= 0 else "?")

            for _, row in df.iterrows():
                rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}

                # City filter (if provided)
                if city_norm:
                    rec_city = normalize_text(rec.get("OriginalCity") or rec.get("City") or "")
                    if city_norm not in rec_city:
                        continue

                # Date filters (if provided)
                appl = rec.get("AppliedDate", "")
                if date_from and appl:
                    if appl < date_from:
                        continue
                if date_to and appl:
                    if appl > date_to:
                        continue

                # Build normalized candidate addresses from record
                rec_addrs = record_address_values(rec)
                if not rec_addrs:
                    continue

                # scoring-based match
                matched = False
                matched_by = ""
                input_norm = normalize_text(input_addr)
                route_tokens = (route_norm or "").split()
                input_main = main_street_part(input_addr)

                # Prefer exact AddressDescription-style canonical equality if available
                rec_addrdesc = rec.get("AddressDescription") or ""
                if rec_addrdesc:
                    rec_num, rec_tokens = canonicalize_address_main(rec_addrdesc)
                    if in_num and rec_num and in_num == rec_num and rec_tokens and in_tokens:
                        if set(rec_tokens) == set(in_tokens):
                            matched = True
                            matched_by = "addrdesc_canonical"

                # helper: count token matches between route tokens and candidate tokens
                def token_match_count(tokens, cand_tokens):
                    c = 0
                    for t in tokens:
                        if not t: 
                            continue
                        for ct in cand_tokens:
                            if ct.startswith(t) or ct == t:
                                c += 1
                                break
                    return c

                for cand in rec_addrs:
                    score = 0

                    # ZIP helps but is not required
                    if zip_from_google or zip_code:
                        z = (zip_from_google or zip_code or "").strip()
                        rec_zip = (rec.get("OriginalZip") or rec.get("ZipCode") or "").strip()
                        if z and rec_zip and rec_zip.startswith(z):
                            score += 3
                        elif z and (cand.endswith(z) or (" " + z + " ") in (" " + cand + " ")):
                            score += 2

                    # street number alignment helps
                    if street_number and (cand.startswith(street_number + " ") or cand.startswith(street_number + "-") or cand.startswith(street_number + ",")):
                        score += 2

                    # route tokens overlap
                    if route_tokens:
                        cand_tokens = cand.split()
                        count = token_match_count(route_tokens, cand_tokens)
                        required = 1 if len(route_tokens) == 1 else min(2, len(route_tokens))
                        if count >= required:
                            score += 2

                    # main-part (before comma) equality/containment
                    cand_main = main_street_part(cand)
                    if input_main and cand_main:
                        if cand_main == input_main:
                            score += 3
                        elif cand_main.startswith(input_main) or input_main.startswith(cand_main):
                            score += 2

                    # substring fallback
                    if input_norm and input_norm in cand:
                        score += 1

                    if score >= 3:
                        matched = True
                        if not matched_by:
                            matched_by = "score"
                        break

                if not matched:
                    continue

                # If matched by address, check optional permit requirement
                if permit:
                    p = (permit or "").strip().lower()
                    rec_p = (rec.get("PermitNum") or rec.get("PermitNumber") or "").strip().lower()
                    if not rec_p or p != rec_p:
                        continue  # address matched but permit did not; try next record

                # Date filters already applied above; if we reached here, address (and permit, if any) matched
                # Generate PDF immediately and return
                rec_id = pick_id_from_record(rec)
                try:
                    pdf_path = generate_pdf_from_template(rec, str(TEMPLATES_DIR / "certificate-placeholder.html"))
                    view_url = f"/view/{rec_id}"
                    download_url = f"/download/{rec_id}.pdf"
                    logging.info("SEARCH early-return PDF generated | id=%s blob=%s", rec_id, name)
                    dur_ms = int((time.perf_counter() - start_t) * 1000)
                    return JSONResponse({
                        "results": [rec],
                        "view_url": view_url,
                        "download_url": download_url,
                        "duration_ms": dur_ms,
                        "canonical_hit": matched_by == "addrdesc_canonical"
                    })
                except HTTPException as he:
                    logging.exception("PDF generation failed during search for id=%s: %s", rec_id, he)
                    dur_ms = int((time.perf_counter() - start_t) * 1000)
                    return JSONResponse({
                        "results": [rec],
                        "pdf_error": str(he.detail),
                        "duration_ms": dur_ms,
                        "canonical_hit": matched_by == "addrdesc_canonical"
                    })
    except Exception as e:
        logging.exception("Search error: %s", e)
        # Return whatever found so far plus error note
        return JSONResponse({"results": results, "error": str(e)})

    # Optionally: sort results by AppliedDate descending if present
    try:
        results_sorted = sorted(results, key=lambda r: r.get("AppliedDate") or "", reverse=True)
    except Exception:
        results_sorted = results

    dur_ms = int((time.perf_counter() - start_t) * 1000)
    logging.info("SEARCH done | results=%d canonical_hit=%s duration_ms=%d", len(results_sorted), canonical_hit, dur_ms)
    return JSONResponse({"results": results_sorted, "duration_ms": dur_ms, "canonical_hit": canonical_hit})

# ---------- NEW: record endpoint now generates PDF only if record exists ----------
@app.get("/record")
def get_record(permit_id: str = Query(...)):
    if not permit_id or not permit_id.strip():
        raise HTTPException(status_code=400, detail="permit_id required")

    logging.info("Searching record for permit_id=%s", permit_id)

    # 1) If a local PDF already exists, return record hint + view/download URLs without requiring Azure
    local_pdf = get_local_pdf_path_if_exists(permit_id)
    if local_pdf:
        logging.info("Found cached PDF on disk for %s (%s); returning view/download URLs", permit_id, local_pdf)
        # best-effort: try to read record from Azure but not required
        try:
            record = _find_record_in_azure_csvs(permit_id) if src_container else None
        except Exception as e:
            logging.debug("Ignoring azure lookup error for get_record local-pdf case: %s", e)
            record = None
        # return minimal JSON so frontend can open view/download
        view_url = f"/view/{permit_id}"
        download_url = f"/download/{permit_id}.pdf"
        return JSONResponse({"record": record or {"id": permit_id}, "view_url": view_url, "download_url": download_url})

    # 2) No local PDF; try to find record in Azure (this is your original behavior)
    if not src_container:
        logging.warning("SRC_CONTAINER not configured and no local PDF for %s", permit_id)
        raise HTTPException(status_code=503, detail="No Azure SRC_CONTAINER configured and no cached PDF available.")

    try:
        record = _find_record_in_azure_csvs(permit_id)
    except Exception as e:
        logging.exception("Azure lookup failed for %s: %s", permit_id, e)
        raise HTTPException(status_code=503, detail="Azure lookup failed. Check server network/credentials.")

    if not record:
        logging.info("Record not found for permit_id=%s", permit_id)
        raise HTTPException(status_code=404, detail="Record not found. Your search data is not in records or the permit number is incorrect.")

    # 3) If found, generate PDF and return URLs (existing flow)
    permit_id_safe = pick_id_from_record(record)
    logging.info("Record found for permit_id=%s — generating PDF view/download urls", permit_id_safe)
    template_path = str(TEMPLATES_DIR / "certificate-placeholder.html")
    try:
        pdf_path = generate_pdf_from_template(record, template_path)
        logging.info("PDF generated at %s for permit %s (record endpoint)", pdf_path, permit_id_safe)
    except HTTPException as he:
        logging.exception("PDF generation failed for %s: %s", permit_id_safe, he)
        return JSONResponse({
            "record": record,
            "pdf_error": str(he.detail),
            "view_url": None,
            "download_url": None
        })

    view_url = f"/view/{permit_id_safe}"
    download_url = f"/download/{permit_id_safe}.pdf"
    return JSONResponse({"record": record, "view_url": view_url, "download_url": download_url})

# ----------------- PDF generation, view and download endpoints -----------------
def generate_pdf_from_template(record: dict, template_path: str) -> str:
    tmpdir = tempfile.gettempdir()
    permit_id = pick_id_from_record(record)
    pdf_file = tmp_pdf_path_for_id(permit_id)
    try:
        logging.info("Starting PDF generation for record %s", permit_id)

        addr1 = record.get("OriginalAddress1", "") or record.get("OriginalAddress", "")
        addr2 = record.get("OriginalAddress2", "")
        property_address = (addr1 + " " + addr2).strip()

        # ensure 'other' exists so templates referencing other.* don't break
        ctx = {
            "dates": {
                "application": record.get("AppliedDate", ""),
                "completion": record.get("CompletedDate", ""),
                "expires": record.get("ExpiresDate", ""),
                "last_updated": record.get("LastUpdated", ""),
                "status_date": record.get("StatusDate", "")
            },
            "permit": {
                "class": record.get("PermitClass", ""),
                "classification": record.get("PermitClassMapped", ""),
                "number": record.get("PermitNum", "") or record.get("PermitNumber", ""),
                "type": record.get("PermitType", ""),
                "type_classification": record.get("PermitTypeMapped", ""),
                "id": permit_id,
                "certificate_number": record.get("certificate_number", "") or record.get("CertificateNo", "") or ""
            },
            "property": {
                "address_description": record.get("AddressDescription", ""),
                "address": property_address,
                "city": record.get("OriginalCity", "") or record.get("City", "") or record.get("PropertyCity", "N/A"),
                "state": record.get("OriginalState") or record.get("State", ""),
                "zip_code": record.get("OriginalZip") or record.get("ZipCode") or "",
                "pin": record.get("PIN", "")
            },
            "status_current": record.get("StatusCurrent", ""),
            "current_status": record.get("StatusCurrentMapped", ""),
            "other": {
                "online_record_url": record.get("Link", "") or "",
                "publisher": record.get("Publisher", "") or ""
            },
            "logo_image_url": str((BASE_DIR / "Medias" / "badge.png").as_uri()) if (BASE_DIR / "Medias" / "badge.png").exists() else "",
            "map_image_url": str((BASE_DIR / "Medias" / "map.png").as_uri()) if (BASE_DIR / "Medias" / "map.png").exists() else "",
            "generated_on_date": datetime.now().strftime("%m/%d/%Y"),
            "record": record
        }

        tmpl_name = Path(template_path).name
        template = templates.get_template(tmpl_name)
        rendered_html = template.render(**ctx)

        # Try WeasyPrint (if available)
        if WEASYPRINT_AVAILABLE:
            try:
                from weasyprint import HTML
                base = str(BASE_DIR)
                HTML(string=rendered_html, base_url=base).write_pdf(pdf_file)
                logging.info("PDF generated via WeasyPrint at %s", pdf_file)
                return pdf_file
            except ImportError as we_import_err:
                logging.warning("WeasyPrint import failed at runtime: %s — trying wkhtmltopdf fallback", we_import_err)
            except Exception as we_err:
                logging.error("WeasyPrint runtime error: %s — trying wkhtmltopdf fallback", we_err, exc_info=True)
        else:
            logging.info("WeasyPrint not available — using wkhtmltopdf fallback")

        # Fallback: wkhtmltopdf
        tmp_html = os.path.join(tmpdir, f"{permit_id}.html")
        with open(tmp_html, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        # common windows default path if not on PATH
        wk_default = "wkhtmltopdf"
        if os.name == "nt":
            common_paths = [
                os.getenv("WKHTMLTOPDF_PATH", ""),
                "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe",
                "C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
            ]
            wk = next((p for p in common_paths if p and os.path.exists(p)), "wkhtmltopdf")
        else:
            wk = os.getenv("WKHTMLTOPDF_PATH", wk_default)
        try:
            subprocess.run([wk, tmp_html, pdf_file], check=True)
            logging.info("PDF generated via wkhtmltopdf at %s", pdf_file)
            return pdf_file
        except Exception as wk_err:
            logging.warning("wkhtmltopdf failed: %s — trying Chrome headless fallback", wk_err)

        # Fallback #2: Chrome headless
        chrome_candidates = []
        if os.name == "nt":
            chrome_candidates = [
                os.getenv("CHROME_PATH", ""),
                "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
            ]
        else:
            chrome_candidates = [
                os.getenv("CHROME_PATH", ""),
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
            ]
        chrome = next((p for p in chrome_candidates if p and os.path.exists(p)), None)
        if chrome:
            try:
                subprocess.run([chrome, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_file}", tmp_html], check=True)
                logging.info("PDF generated via Chrome headless at %s", pdf_file)
                return pdf_file
            except Exception as ch_err:
                logging.exception("Chrome headless PDF failed: %s", ch_err)

        # No renderer worked
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF generation failed: WeasyPrint not available, wkhtmltopdf not found/failed, "
                "and Chrome headless fallback failed. Please install one renderer (WeasyPrint, wkhtmltopdf, or Chrome)."
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("PDF generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

@app.get("/view/{permit_id}")
def view_pdf(permit_id: str):
    logging.info("View request for permit %s", permit_id)

    # 1) Serve cached file if present (no Azure required)
    local_pdf = get_local_pdf_path_if_exists(permit_id)
    if local_pdf:
        logging.info("Serving cached PDF for %s from %s", permit_id, local_pdf)
        # serve inline — do NOT use filename parameter (that can trigger attachment behavior)
        return FileResponse(
            local_pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=\"{os.path.basename(local_pdf)}\""}
        )

    # 2) If no local PDF, try to find record and generate (requires Azure)
    if not src_container:
        logging.info("No cached PDF and no SRC_CONTAINER configured for view %s", permit_id)
        raise HTTPException(status_code=404, detail="Certificate PDF not found (no cached PDF and storage unavailable).")

    record = _find_record_in_azure_csvs(permit_id)
    if not record:
        logging.info("Record %s not found for view", permit_id)
        raise HTTPException(status_code=404, detail="Certificate PDF not found")

    # generate then serve
    pdf_file = tmp_pdf_path_for_id(permit_id)
    if not os.path.exists(pdf_file):
        logging.info("PDF not on disk for %s — generating now", permit_id)
        pdf_file = generate_pdf_from_template(record, str(TEMPLATES_DIR / "certificate-placeholder.html"))

    logging.info("Serving PDF %s for permit %s (inline view)", pdf_file, permit_id)
    return FileResponse(
        pdf_file,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{os.path.basename(pdf_file)}\""}
    )

@app.get("/download/{permit_id}")
@app.get("/download/{permit_id}.pdf")
def download_pdf(permit_id: str):
    logging.info("Download request for permit %s", permit_id)

    # 1) Serve cached if present
    local_pdf = get_local_pdf_path_if_exists(permit_id)
    if local_pdf:
        logging.info("Serving cached download for %s from %s", permit_id, local_pdf)
        return FileResponse(local_pdf, media_type="application/pdf", filename=f"{permit_id}.pdf")

    # 2) Otherwise try to generate (requires Azure)
    if not src_container:
        raise HTTPException(status_code=404, detail="PDF not found (no cached PDF and storage unavailable)")

    record = _find_record_in_azure_csvs(permit_id)
    if not record:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_file = tmp_pdf_path_for_id(permit_id)
    if not os.path.exists(pdf_file):
        logging.info("PDF not on disk for download %s — generating now", permit_id)
        pdf_file = generate_pdf_from_template(record, str(TEMPLATES_DIR / "certificate-placeholder.html"))

    return FileResponse(pdf_file, media_type="application/pdf", filename=f"{pick_id_from_record(record)}.pdf")

# ----------------- Root fallback -----------------
@app.get("/")
def root():
    return JSONResponse({"service": "permitvista api", "health": "/health"})
# ----------------- Run server (dev) -----------------
if __name__ == "__main__":
    host, port = "127.0.0.1", int(os.getenv("PORT", "8000"))
    url = f"http://{host}:{port}/"
    threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=host, port=port)
