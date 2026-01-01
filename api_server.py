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
import secrets
from collections import defaultdict
from datetime import datetime, timedelta

# ----------------- Address Index Import -----------------
from address_index import AddressIndex, set_extraction_functions

# ----------------- Token Management for Privacy -----------------
# Store token -> permit_id mapping (with expiration)
_token_store: dict[str, dict] = {}  # token -> {"permit_id": str, "expires_at": datetime}
_token_lock = threading.Lock()

def generate_secure_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)  # 32 bytes = 43 characters URL-safe

def create_token_for_permit(permit_id: str) -> str:
    """Create a token for a permit_id and store it (expires in 24 hours)"""
    token = generate_secure_token()
    expires_at = datetime.now() + timedelta(hours=24)
    with _token_lock:
        _token_store[token] = {
            "permit_id": permit_id,
            "expires_at": expires_at
        }
        # Clean up expired tokens (keep last 1000)
        if len(_token_store) > 1000:
            now = datetime.now()
            expired = [t for t, data in _token_store.items() if data["expires_at"] < now]
            for t in expired[:500]:  # Remove up to 500 expired tokens
                _token_store.pop(t, None)
    return token

def get_permit_id_from_token(token: str) -> Optional[str]:
    """Get permit_id from token, return None if invalid or expired"""
    with _token_lock:
        data = _token_store.get(token)
        if not data:
            return None
        if data["expires_at"] < datetime.now():
            _token_store.pop(token, None)  # Remove expired token
            return None
        return data["permit_id"]

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
SAMPLE_PERMIT_LIMIT = int(os.getenv("SAMPLE_PERMIT_LIMIT", "1000"))
MAX_CSV_FILES = int(os.getenv("MAX_CSV_FILES", "100"))
# Enable/disable indexing (set to "false" to disable indexing and save memory)
ENABLE_INDEXING = os.getenv("ENABLE_INDEXING", "true").lower() == "true"
# Limit index to recent files only (set to number of files, e.g., "10000" for last 10k files)
INDEX_FILE_LIMIT = int(os.getenv("INDEX_FILE_LIMIT", "0"))  # 0 = index all files

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

# ----------------- Address Index Initialization -----------------
address_index = AddressIndex()
# Set extraction functions for the index (called after functions are defined)
_index_initialized = False

def _initialize_address_index():
    """Initialize address index with extraction functions and load from Azure if available"""
    global _index_initialized
    logging.info("🔧 Initializing address index system...")
    if not _index_initialized:
        set_extraction_functions(
            extract_record_address_components,
            normalize_text,
            pick_id_from_record
        )
        _index_initialized = True
        logging.info("✅ Address index extraction functions registered")
        
        # Set Azure storage for index persistence (use pdf_container)
        if pdf_container:
            address_index.set_azure_storage(pdf_container)
            logging.info("✅ Address index will use Azure Storage for persistence: %s/%s", 
                       OUTPUT_CONTAINER, address_index.INDEX_BLOB_NAME)
            # Try to load existing index from Azure on startup
            if address_index.load_from_azure():
                logging.info("✅ Address index loaded from Azure Storage on startup - ready for fast searches!")
            else:
                logging.info("ℹ️ No existing index found in Azure - will build and save on first search")
        else:
            logging.warning("⚠️ Azure storage (pdf_container) not available - indexing disabled (ENABLE_INDEXING will be ignored)")
    else:
        logging.info("Address index already initialized")

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

def extract_record_address_components(rec: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (street_number, street_name, zip_code) from a record.
    Uses OriginalAddress1, AddressDescription, and OriginalZip fields from CSV.
    Returns normalized values for matching.
    """
    # Extract street number from address fields
    street_number = None
    addr_fields = ["OriginalAddress1", "AddressDescription", "Address", "OriginalAddress", "StreetAddress", "PropertyAddress"]
    for field in addr_fields:
        val = rec.get(field)
        if val and isinstance(val, str) and val.strip():
            num_match = NUM_RE.match(str(val).strip())
            if num_match:
                street_number = num_match.group(1).strip()
                break
    
    # Extract street name - try OriginalAddress1 first (more reliable, AddressDescription might be a number)
    street_name = None
    orig_addr = rec.get("OriginalAddress1") or ""
    if orig_addr:
        # Format is usually: "3110 Chapin Ave W " or "3613 Lindell Ave  "
        cleaned = str(orig_addr).strip()
        # Remove street number if present
        if street_number and cleaned.startswith(street_number):
            cleaned = cleaned[len(street_number):].strip()
        # Remove direction tokens
        cleaned = re.sub(r"^\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
        # Remove street type
        cleaned = re.sub(r"\b(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Terrace|Ter|Place|Pl|Way|Wy|Circle|Cir|Trail|Trl|Parkway|Pkwy|Square|Sq)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = normalize_text(cleaned)
        if cleaned:
            street_name = cleaned
    
    # If not found in OriginalAddress1, try AddressDescription (but skip if it's just a number)
    if not street_name:
        addr_desc = rec.get("AddressDescription") or ""
        if addr_desc:
            addr_desc_str = str(addr_desc).strip()
            # Skip if it's just a number (like "1110.0" or "1110")
            if not (addr_desc_str.replace(".", "").replace("-", "").isdigit()):
                # Format is usually: "3110 W Chapin Ave (ACA)" or "3613 Lindell Ave (ACA)"
                # Remove (ACA) tag
                cleaned = re.sub(r"\s*\(aca\)\s*$", "", addr_desc_str, flags=re.IGNORECASE).strip()
                # Remove street number if present
                if street_number and cleaned.startswith(street_number):
                    cleaned = cleaned[len(street_number):].strip()
                # Remove direction tokens (N, S, E, W, etc.) - can be at start
                cleaned = re.sub(r"^\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?\s*", "", cleaned, flags=re.IGNORECASE).strip()
                cleaned = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
                # Remove street type
                cleaned = re.sub(r"\b(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Terrace|Ter|Place|Pl|Way|Wy|Circle|Cir|Trail|Trl|Parkway|Pkwy|Square|Sq)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
                cleaned = normalize_text(cleaned)
                if cleaned:
                    street_name = cleaned
        orig_addr = rec.get("OriginalAddress1") or ""
        if orig_addr:
            # Format is usually: "3110 Chapin Ave W " or "3613 Lindell Ave  "
            cleaned = str(orig_addr).strip()
            # Remove street number if present
            if street_number and cleaned.startswith(street_number):
                cleaned = cleaned[len(street_number):].strip()
            # Remove direction tokens
            cleaned = re.sub(r"^\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"\b(N|S|E|W|NE|NW|SE|SW|North|South|East|West)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
            # Remove street type
            cleaned = re.sub(r"\b(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Terrace|Ter|Place|Pl|Way|Wy|Circle|Cir|Trail|Trl|Parkway|Pkwy|Square|Sq)\b\.?$", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = normalize_text(cleaned)
            if cleaned:
                street_name = cleaned
    
    # Extract zip code
    zip_code = None
    zip_val = rec.get("OriginalZip") or rec.get("ZipCode") or rec.get("ZIP") or rec.get("Zip") or ""
    if zip_val:
        # Handle both string and numeric zip codes (e.g., 33603.0)
        zip_str = str(zip_val).strip()
        # Remove decimal part if present (e.g., "33603.0" -> "33603")
        if "." in zip_str and zip_str.replace(".", "").isdigit():
            zip_str = zip_str.split(".")[0]
        zip_match = ZIP_RE.search(zip_str)
        if zip_match:
            zip_code = zip_match.group(1).strip()
    
    return street_number, street_name, zip_code

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

# Initialize address index after all helper functions are defined
_initialize_address_index()

# ----------------- Endpoints -----------------
@app.get("/index-status")
def index_status():
    """Get current status of the address index"""
    stats = address_index.get_stats()
    total_files = 55043  # Approximate total CSV files
    if stats["loading"]:
        progress_pct = (stats["total_files"] / total_files * 100) if total_files > 0 else 0
        status_msg = f"Indexing in progress: {stats['total_files']}/{total_files} files ({progress_pct:.1f}%)"
    elif stats["loaded"]:
        status_msg = f"Index complete: {stats['total_files']} files, {stats['total_records']} records, {stats['unique_addresses']} unique addresses"
    else:
        status_msg = "Index not started"
    
    return JSONResponse({
        "status": "loading" if stats["loading"] else ("loaded" if stats["loaded"] else "not_started"),
        "message": status_msg,
        "stats": stats,
        "total_files_estimated": total_files
    })

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
    canonical_hit = True

    try:
        # If user provided date range, use scan_limit for performance
        # If no date range, scan ALL files until exact match found
        has_date_range = bool(date_from or date_to)
        if has_date_range:
            scan_max = min(max(1, int(scan_limit or MAX_CSV_FILES)), 5000)
            logging.info("SEARCH date range provided - using scan_limit=%d", scan_max)
        else:
            # No date range - scan ALL files until match found
            scan_max = 999999  # Effectively unlimited
            logging.info("SEARCH no date range - will scan ALL files until exact match found")
        
        # List all CSV files available in Azure (needed for both index and scanning)
        all_csv_files = []
        for blob in src_container.list_blobs():
            name = getattr(blob, "name", str(blob))
            if name.lower().endswith(".csv"):
                all_csv_files.append(name)
        
        # Check if we should use index: no date range + structured fields + index available
        user_provided_structured = bool(street_number_q or street_name_q or zip_q)
        
        # Debug: Log index status
        logging.info("SEARCH index check | ENABLE_INDEXING=%s has_date_range=%s user_provided_structured=%s is_loading=%s has_azure_storage=%s is_loaded=%s index_size=%d",
                    ENABLE_INDEXING, has_date_range, user_provided_structured, 
                    address_index.is_loading(), address_index._use_azure_storage, 
                    address_index.is_loaded(), len(address_index._index))
        
        # First, try to ensure index is loaded if it exists in Azure
        # Only if indexing is enabled and Azure storage is available
        if (ENABLE_INDEXING and not has_date_range and user_provided_structured
            and not address_index.is_loading() and src_container and address_index._use_azure_storage):
            # Check if index is loaded AND has data in memory
            index_has_data = address_index.is_loaded() and len(address_index._index) > 0
            
            if not index_has_data:
                # Try to load from Azure (synchronously, so we can use it for this search)
                logging.info("SEARCH index not in memory - attempting to load from Azure")
                if address_index.load_from_azure():
                    logging.info("✅ Index loaded from Azure - will use for this search")
                    index_has_data = len(address_index._index) > 0
                else:
                    # Index doesn't exist in Azure - trigger background build for future searches
                    logging.info("SEARCH index not found in Azure - triggering background index build")
                    def build_index_background():
                        try:
                            address_index.build_index_from_azure(src_container, max_files=INDEX_FILE_LIMIT if INDEX_FILE_LIMIT > 0 else None)
                            logging.info("✅ Index build complete and saved to Azure - future searches will be fast!")
                        except Exception as e:
                            logging.exception("Background index build failed: %s", e)
                    threading.Thread(target=build_index_background, daemon=True).start()
        else:
            # Log why index loading is being skipped
            if not ENABLE_INDEXING:
                logging.info("SEARCH index loading skipped: ENABLE_INDEXING=false")
            elif has_date_range:
                logging.info("SEARCH index loading skipped: date range provided")
            elif not user_provided_structured:
                logging.info("SEARCH index loading skipped: no structured address fields provided")
            elif address_index.is_loading():
                logging.info("SEARCH index loading skipped: index is currently being built")
            elif not src_container:
                logging.info("SEARCH index loading skipped: src_container not available")
            elif not address_index._use_azure_storage:
                logging.info("SEARCH index loading skipped: Azure storage not configured for index")
        
        # Now check if we can use the index
        index_has_data = address_index.is_loaded() and len(address_index._index) > 0
        use_index = (ENABLE_INDEXING and not has_date_range and user_provided_structured and index_has_data)
        
        if not use_index and ENABLE_INDEXING and not has_date_range and user_provided_structured:
            if not address_index._use_azure_storage:
                logging.info("SEARCH Azure storage not available - indexing disabled - using full scan")
            elif address_index.is_loading():
                logging.info("SEARCH index is currently being built - using full scan for this request")
            else:
                logging.info("SEARCH index not available - using full scan (index may be building in background)")
        
        # ============================================================
        # INDEX-BASED SEARCH (fast path when index is loaded from Azure)
        # ============================================================
        if use_index:
            logging.info("SEARCH ✅ USING INDEX - Fast lookup mode (index has %d unique addresses, %d total records)", 
                       len(address_index._index), address_index._total_records_indexed)
            
            # Extract user's address components (same as structured matching logic)
            user_street_number = (street_number_q or "").strip() if street_number_q else None
            user_street_name = normalize_text(street_name_q or "") if street_name_q else None
            user_zip = (zip_q or "").strip() if zip_q else None
            
            # Look up in index
            index_matches = address_index.lookup(
                street_number=user_street_number,
                street_name=user_street_name,
                zip_code=user_zip,
                date_from=date_from,
                date_to=date_to,
                permit=permit,
                city=city
            )
            
            logging.info("SEARCH index lookup found %d candidate records", len(index_matches))
            
            # Process each match from index (works for both full and partial index)
            for location in index_matches:
                try:
                    # Read the specific CSV file and row
                    df = _read_csv_bytes_from_blob(src_container, location.blob_name)
                    if df is None:
                        continue
                    
                    # Get the specific row
                    if location.row_index >= len(df):
                        continue
                    
                    row = df.iloc[location.row_index]
                    rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                    
                    # Apply same matching logic as scanning (double-check address match)
                    rec_street_number, rec_street_name, rec_zip = extract_record_address_components(rec)
                    
                    address_matched = True
                    
                    # Street number check
                    if user_street_number:
                        if not rec_street_number or user_street_number != rec_street_number:
                            address_matched = False
                    
                    # Street name check
                    if address_matched and user_street_name:
                        if not rec_street_name or user_street_name.strip() != rec_street_name.strip():
                            address_matched = False
                    
                    # ZIP check
                    if address_matched and user_zip:
                        user_zip_clean = user_zip[:5]
                        if not rec_zip or user_zip_clean != rec_zip[:5]:
                            address_matched = False
                    
                    if not address_matched:
                        continue
                    
                    # City filter (if provided)
                    if city_norm:
                        rec_city = normalize_text(rec.get("OriginalCity") or rec.get("City") or "")
                        if city_norm not in rec_city:
                            continue
                    
                    # Permit filter (if provided)
                    if permit:
                        p = (permit or "").strip().lower()
                        rec_p = (rec.get("PermitNum") or rec.get("PermitNumber") or "").strip().lower()
                        if not rec_p or p != rec_p:
                            continue
                    
                    # Date filter (if provided - already filtered by index, but double-check)
                    appl = rec.get("AppliedDate", "")
                    if date_from and appl and appl < date_from:
                        continue
                    if date_to and appl and appl > date_to:
                        continue
                    
                    # All checks passed - generate PDF and return
                    rec_id = pick_id_from_record(rec)
                    try:
                        pdf_path = generate_pdf_from_template(rec, str(TEMPLATES_DIR / "certificate-placeholder.html"))
                        token = create_token_for_permit(rec_id)
                        view_url = f"/view/{token}"
                        download_url = f"/download/{token}.pdf"
                        logging.info("SEARCH index match - PDF generated | id=%s blob=%s token=%s", rec_id, location.blob_name, token[:8])
                        dur_ms = int((time.perf_counter() - start_t) * 1000)
                        return JSONResponse({
                            "results": [rec],
                            "view_url": view_url,
                            "download_url": download_url,
                            "duration_ms": dur_ms,
                            "canonical_hit": True,
                            "index_used": True
                        })
                    except HTTPException as he:
                        logging.exception("PDF generation failed during index search for id=%s: %s", rec_id, he)
                        dur_ms = int((time.perf_counter() - start_t) * 1000)
                        return JSONResponse({
                            "results": [rec],
                            "pdf_error": str(he.detail),
                            "duration_ms": dur_ms,
                            "canonical_hit": True,
                            "index_used": True
                        })
                
                except Exception as e:
                    logging.warning("Error processing index match from %s row %d: %s", location.blob_name, location.row_index, e)
                    continue
            
            # If using index and no matches found, return "not found"
            dur_ms = int((time.perf_counter() - start_t) * 1000)
            logging.info("SEARCH index lookup completed - no matches found | duration_ms=%d", dur_ms)
            return JSONResponse({
                "results": [],
                "message": "Record not found. Your search data is not in records or the provided fields do not match exactly.",
                "duration_ms": dur_ms,
                "canonical_hit": True,
                "index_used": True
            })
        
        # ============================================================
        # SCANNING-BASED SEARCH (fallback when index not available)
        # ============================================================
        # Note: all_csv_files is already defined above
        
        scanned = 0  # Initialize scanned counter for scanning loop
        
        # Extract date from filename or path for filtering and sorting
        # Files are typically in format: YYYY/MM/DD/tampa_permits_YYYYMMDD_*.csv
        def extract_date_from_path(path):
            """Extract date from path like '2023/05/05/tampa_permits_20230505_*.csv'
            Returns (year, month, day) tuple or (0, 0, 0) if not found"""
            parts = path.split('/')
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    return (year, month, day)
                except (ValueError, IndexError):
                    pass
            # Fallback: try to extract from filename
            import re
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', path)
            if date_match:
                return (int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
            return (0, 0, 0)  # Put files without dates at the end
        
        def date_tuple_to_string(date_tuple):
            """Convert (year, month, day) to 'YYYY-MM-DD' string"""
            year, month, day = date_tuple
            if year == 0:
                return None
            return f"{year:04d}-{month:02d}-{day:02d}"
        
        # If user provided date range, filter CSV files to only those within the range
        if date_from or date_to:
            logging.info("SEARCH date range provided | date_from=%s date_to=%s - filtering CSV files", date_from, date_to)
            filtered_csv_files = []
            for csv_file in all_csv_files:
                file_date_tuple = extract_date_from_path(csv_file)
                file_date_str = date_tuple_to_string(file_date_tuple)
                
                if file_date_str is None:
                    # File has no date - include it (better safe than sorry)
                    filtered_csv_files.append(csv_file)
                    continue
                
                # Check if file date is within range
                include_file = True
                if date_from and file_date_str < date_from:
                    include_file = False
                if date_to and file_date_str > date_to:
                    include_file = False
                
                if include_file:
                    filtered_csv_files.append(csv_file)
            
            original_count = len(all_csv_files)
            all_csv_files = filtered_csv_files
            logging.info("SEARCH date range filter | filtered from %d to %d CSV files", original_count, len(all_csv_files))
        
        # Sort by date descending (newest first) - this ensures recent records are found first
        all_csv_files.sort(key=extract_date_from_path, reverse=True)
        
        logging.info("SEARCH Azure scan | total_csv_files=%d scan_limit=%d", len(all_csv_files), scan_max)
        if len(all_csv_files) > 0 and len(all_csv_files) <= 20:
            logging.info("SEARCH CSV files in Azure (newest first): %s", all_csv_files[:20])
        elif len(all_csv_files) > 20:
            logging.info("SEARCH CSV files in Azure (newest first, first 20): %s", all_csv_files[:20])
            logging.info("SEARCH ... and %d more CSV files", len(all_csv_files) - 20)
        
        # Now scan the CSV files (already sorted newest first)
        # If no date range provided, scan ALL files until match found (no limit)
        # If date range provided, respect scan_max limit
        files_to_scan = all_csv_files if not (date_from or date_to) else all_csv_files[:scan_max]
        actual_scan_limit = len(all_csv_files) if not (date_from or date_to) else scan_max
        
        if not (date_from or date_to):
            logging.info("SEARCH no date range provided - will scan ALL %d files until match found", len(all_csv_files))
        else:
            logging.info("SEARCH date range provided - will scan up to %d files", scan_max)
        
        scanned_file_names = set()  # Track which files we've scanned
        for csv_file_name in files_to_scan:
            if csv_file_name in scanned_file_names:
                continue
            scanned_file_names.add(csv_file_name)
            
            # Get blob by name
            try:
                blob_client = src_container.get_blob_client(csv_file_name)
                if not blob_client.exists():
                    logging.warning("CSV file not found in Azure: %s", csv_file_name)
                    continue
            except Exception as e:
                logging.warning("Error checking blob %s: %s", csv_file_name, e)
                continue
            
            name = csv_file_name
            # Only enforce scan limit if date range was provided
            if (date_from or date_to) and scanned >= scan_max:
                logging.info("Reached scan limit during search (%d)", scan_max)
                break

            df = _read_csv_bytes_from_blob(src_container, name)
            scanned += 1
            if df is None:
                continue
            try:
                df_len = len(df)
            except Exception:
                df_len = -1
            
            # Log progress every 100 files (to avoid log spam when scanning all files)
            if scanned % 100 == 0 or scanned <= 20:
                total_to_scan = len(files_to_scan) if not (date_from or date_to) else scan_max
                logging.info("Scanning blob %d/%s: %s rows=%s", scanned, total_to_scan if (date_from or date_to) else "ALL", name, df_len if df_len >= 0 else "?")

            for _, row in df.iterrows():
                rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}

                # City filter (if provided) - OK to check early
                if city_norm:
                    rec_city = normalize_text(rec.get("OriginalCity") or rec.get("City") or "")
                    if city_norm not in rec_city:
                        continue

                # ============================================================
                # STEP 1: ADDRESS MATCHING (REQUIRED - must match exactly)
                # ============================================================
                user_provided_structured = bool(street_number_q or street_name_q or zip_q)
                address_matched = False
                matched_by = ""
                
                if user_provided_structured:
                    # STRICT FIELD-BASED MATCHING: All provided fields must match exactly
                    user_street_number = (street_number_q or "").strip() if street_number_q else None
                    user_street_name = normalize_text(street_name_q or "") if street_name_q else None
                    user_zip = (zip_q or "").strip() if zip_q else None

                    rec_street_number, rec_street_name, rec_zip = extract_record_address_components(rec)

                    address_matched = True
                    matched_by = "field_match"

                    # 1️⃣ Street number check (EXACT match required if provided)
                    if user_street_number:
                        if not rec_street_number or user_street_number != rec_street_number:
                            address_matched = False
                            logging.debug(f"Street number mismatch: user='{user_street_number}' vs rec='{rec_street_number}'")

                    # 2️⃣ Street name check (EXACT match required if provided)
                    if address_matched and user_street_name:
                        if not rec_street_name:
                            address_matched = False
                            logging.debug("Street name check failed: record has no street name")
                        else:
                            # EXACT matching: normalized street names must be identical
                            # No partial/prefix matching - must match exactly
                            # Compare as strings (already normalized)
                            if user_street_name.strip() != rec_street_name.strip():
                                address_matched = False
                                logging.debug(f"[MATCH] Street name EXACT mismatch: user='{user_street_name}' vs rec='{rec_street_name}' (blob={name})")
                            else:
                                logging.debug(f"[MATCH] Street name matched: '{user_street_name}' == '{rec_street_name}' (blob={name})")

                    # 3️⃣ ZIP validation (EXACT match required if provided)
                    if address_matched and user_zip:
                        user_zip_clean = user_zip[:5]
                        if not rec_zip or user_zip_clean != rec_zip[:5]:
                            address_matched = False
                            logging.debug(f"[MATCH] ZIP mismatch: user='{user_zip_clean}' vs rec='{rec_zip[:5] if rec_zip else None}' (blob={name})")
                        else:
                            logging.debug(f"[MATCH] ZIP matched: '{user_zip_clean}' == '{rec_zip[:5]}' (blob={name})")

                    # If structured fields provided but don't match, skip this record (no fallback)
                    if not address_matched:
                        logging.debug(f"[MATCH] Address mismatch - skipping record (blob={name}, OriginalAddress1={rec.get('OriginalAddress1', 'N/A')[:50]})")
                        continue
                    else:
                        logging.info(f"[MATCH] Address matched! (blob={name}, OriginalAddress1={rec.get('OriginalAddress1', 'N/A')[:50]})")

                else:
                    # No structured fields provided - use fallback scoring-based matching
                    # Build normalized candidate addresses from record
                    rec_addrs = record_address_values(rec)
                    if not rec_addrs:
                        continue

                    # scoring-based match
                    input_norm = normalize_text(input_addr)
                    route_tokens = (route_norm or "").split()
                    input_main = main_street_part(input_addr)

                    # Prefer exact AddressDescription-style canonical equality if available
                    rec_addrdesc = rec.get("AddressDescription") or ""
                    if rec_addrdesc:
                        rec_num, rec_tokens = canonicalize_address_main(rec_addrdesc)
                        if in_num and rec_num and in_num == rec_num and rec_tokens and in_tokens:
                            if set(rec_tokens) == set(in_tokens):
                                address_matched = True
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
                            address_matched = True
                            if not matched_by:
                                matched_by = "score"
                            break

                # If address doesn't match, skip this record
                if not address_matched:
                    continue

                # ============================================================
                # STEP 2: PERMIT NUMBER MATCHING (if provided, must match exactly)
                # ============================================================
                if permit:
                    p = (permit or "").strip().lower()
                    rec_p = (rec.get("PermitNum") or rec.get("PermitNumber") or "").strip().lower()
                    if not rec_p or p != rec_p:
                        # Address matched but permit did not - skip this record
                        logging.debug(f"Permit mismatch: user='{p}' vs rec='{rec_p}'")
                        continue

                # ============================================================
                # STEP 3: DATE RANGE MATCHING (if provided, must match exactly)
                # ============================================================
                # Date filters are checked AFTER address and permit match
                appl = rec.get("AppliedDate", "")
                if date_from and appl:
                    if appl < date_from:
                        logging.debug(f"Date range mismatch: AppliedDate='{appl}' < date_from='{date_from}'")
                        continue
                if date_to and appl:
                    if appl > date_to:
                        logging.debug(f"Date range mismatch: AppliedDate='{appl}' > date_to='{date_to}'")
                        continue

                # ============================================================
                # STEP 4: ALL CHECKS PASSED - Generate PDF
                # ============================================================
                # If we reached here, address matched (and permit/date if provided)
                # Generate PDF immediately and return
                rec_id = pick_id_from_record(rec)
                try:
                    pdf_path = generate_pdf_from_template(rec, str(TEMPLATES_DIR / "certificate-placeholder.html"))
                    # Use token instead of permit_id in URL for privacy
                    token = create_token_for_permit(rec_id)
                    view_url = f"/view/{token}"
                    download_url = f"/download/{token}.pdf"
                    logging.info("SEARCH early-return PDF generated | id=%s blob=%s token=%s", rec_id, name, token[:8])
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

    # Add view_url and download_url (with tokens) to all results for privacy
    for rec in results_sorted:
        if "view_url" not in rec and "download_url" not in rec:
            rec_id = pick_id_from_record(rec)
            if rec_id:
                token = create_token_for_permit(rec_id)
                rec["view_url"] = f"/view/{token}"
                rec["download_url"] = f"/download/{token}.pdf"

    dur_ms = int((time.perf_counter() - start_t) * 1000)
    logging.info("SEARCH done | results=%d canonical_hit=%s duration_ms=%d", len(results_sorted), canonical_hit, dur_ms)
    
    # If no results found, return "record not found" message
    if len(results_sorted) == 0:
        return JSONResponse({
            "results": [],
            "message": "Record not found. Your search data is not in records or the provided fields do not match exactly.",
            "duration_ms": dur_ms,
            "canonical_hit": canonical_hit
        })
    
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
        # return minimal JSON so frontend can open view/download (use token for privacy)
        token = create_token_for_permit(permit_id)
        view_url = f"/view/{token}"
        download_url = f"/download/{token}.pdf"
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

    # Use token instead of permit_id in URL for privacy
    token = create_token_for_permit(permit_id_safe)
    view_url = f"/view/{token}"
    download_url = f"/download/{token}.pdf"
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

@app.get("/view/{token}")
def view_pdf(token: str):
    # Look up permit_id from token
    permit_id = get_permit_id_from_token(token)
    if not permit_id:
        logging.warning("Invalid or expired token for view request: %s", token[:8])
        raise HTTPException(status_code=404, detail="Invalid or expired link. Please search again.")
    
    logging.info("View request for permit %s (token: %s)", permit_id, token[:8])

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

@app.get("/download/{token}")
@app.get("/download/{token}.pdf")
def download_pdf(token: str):
    # Look up permit_id from token
    permit_id = get_permit_id_from_token(token)
    if not permit_id:
        logging.warning("Invalid or expired token for download request: %s", token[:8])
        raise HTTPException(status_code=404, detail="Invalid or expired link. Please search again.")
    
    logging.info("Download request for permit %s (token: %s)", permit_id, token[:8])

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
