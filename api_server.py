# api_server.py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
import random
import logging
import subprocess
import time
import threading
import webbrowser
from typing import Optional, List, Tuple
import re
import secrets
import json
from collections import defaultdict
from datetime import datetime, timedelta, date

# ----------------- Database Import -----------------
import pyodbc
from queue import Queue, Empty
from contextlib import contextmanager

# ----------------- Shovels API Import -----------------
from shovels_api import get_permits_for_address, ShovelsAPIError

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

# ----------------- Shovels API Configuration -----------------
SHOVELS_API_KEY = os.getenv("SHOVELS_API_KEY", "")
USE_SHOVELS_API = bool(SHOVELS_API_KEY)  # Use Shovels API if key is provided, else fallback to SQL

# ----------------- Database Configuration -----------------
DB_SERVER = os.getenv("DB_SERVER", "permitvista-db.database.windows.net")
DB_DATABASE = os.getenv("DB_DATABASE", "free-sql-db-7590410")
DB_USER = os.getenv("DB_USER", "permitvistaadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

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

# ----------------- Database Connection Pool -----------------
# Connection pool settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_MAX_OVERFLOW", "10"))
DB_CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "60"))
DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
DB_RETRY_DELAY = float(os.getenv("DB_RETRY_DELAY", "1.0"))


# Thread-safe connection pool
_db_pool: Optional[Queue] = None
_db_pool_lock = threading.Lock()
_db_connection_string: Optional[str] = None

def _build_connection_string() -> str:
    """Build the database connection string"""
    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD environment variable is required")
    
    return (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout={DB_CONNECTION_TIMEOUT};"
        f"Command Timeout={DB_QUERY_TIMEOUT};"
    )

def _init_connection_pool():
    """Initialize the connection pool"""
    global _db_pool, _db_connection_string
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_connection_string = _build_connection_string()
                _db_pool = Queue(maxsize=DB_POOL_SIZE)
                logging.info("Initialized database connection pool (size=%d)", DB_POOL_SIZE)

def _create_new_connection():
    """Create a new database connection with retry logic"""
    global _db_connection_string
    if not _db_connection_string:
        _db_connection_string = _build_connection_string()
    
    last_error = None
    for attempt in range(DB_MAX_RETRIES):
        try:
            conn = pyodbc.connect(_db_connection_string, timeout=DB_CONNECTION_TIMEOUT)
            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return conn
        except pyodbc.Error as e:
            last_error = e
            if attempt < DB_MAX_RETRIES - 1:
                delay = DB_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logging.warning("Connection attempt %d/%d failed: %s. Retrying in %.1fs...", 
                              attempt + 1, DB_MAX_RETRIES, str(e), delay)
                time.sleep(delay)
            else:
                logging.error("All %d connection attempts failed. Last error: %s", DB_MAX_RETRIES, e)
    
    raise last_error

def _is_connection_alive(conn) -> bool:
    """Check if a connection is still alive"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return True
    except:
        return False

def _warm_up_connection(conn):
    """Warm up connection by running a simple query to avoid cold start delay (serverless DB optimization)"""
    try:
        cursor = conn.cursor()
        # Run a simple indexed query to warm up the connection and avoid cold start
        cursor.execute("SELECT TOP 1 SearchAddress FROM dbo.permits WHERE SearchAddress IS NOT NULL")
        cursor.fetchone()
        cursor.close()
        logging.debug("Connection warmed up successfully")
    except Exception as e:
        # Non-critical - connection warm-up failure shouldn't break the app
        logging.debug("Connection warm-up failed (non-critical): %s", str(e))


@contextmanager
def get_db_connection():
    """
    Get a database connection from the pool (context manager).
    Returns connection and ensures it's returned to pool or closed on error.
    """
    _init_connection_pool()
    conn = None
    
    try:
        # Try to get connection from pool (non-blocking)
        try:
            conn = _db_pool.get_nowait()
            # Verify connection is still alive
            if not _is_connection_alive(conn):
                try:
                    conn.close()
                except:
                    pass
                conn = None
        except Empty:
            pass
        
        # If no connection from pool, create new one
        if conn is None:
            conn = _create_new_connection()
            # Warm up connection to avoid cold start delay
            _warm_up_connection(conn)
        
        yield conn
        
    except Exception as e:
        # On error, close the connection (don't return to pool)
        if conn:
            try:
                conn.close()
            except:
                pass
        raise
    else:
        # On success, return connection to pool if there's room
        try:
            _db_pool.put_nowait(conn)
        except:
            # Pool is full, close the connection
            try:
                conn.close()
            except:
                pass

def get_table_name(city: Optional[str]) -> List[str]:
    """Determine which table(s) to query based on city parameter"""
    city_lower = (city or "").strip().lower()
    
    if "tampa" in city_lower:
        return ["dbo.permits"]
    elif "miami" in city_lower:
        return ["dbo.miami_permits"]
    elif "orlando" in city_lower:
        return ["dbo.orlando_permits"]
    else:
        # If no city specified, search all tables
        return ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]

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

# Database connection test on startup
try:
    if DB_PASSWORD:
        with get_db_connection() as test_conn:
            # Connection is automatically returned to pool
            pass
        logging.info("✅ Database connection successful: %s/%s", DB_SERVER, DB_DATABASE)
    else:
        logging.warning("⚠️ DB_PASSWORD not set - database queries will fail")
except Exception as e:
    logging.warning("⚠️ Database connection test failed: %s - queries will fail at runtime", e)

# ----------------- Endpoints -----------------

@app.get("/health")
def health():
    return JSONResponse({
        "azure_connected": bool(src_container),
        "src_container": SRC_CONTAINER,
        "output_container": OUTPUT_CONTAINER
    })

@app.get("/db-info")
def db_info():
    """Get database information including indexes"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            tables = ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]
            db_info = {"tables": {}, "indexes": {}}

            for table in tables:
                try:
                    # Get table info
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    db_info["tables"][table] = {"record_count": count}

                    # Get indexes for address columns
                    address_cols = ["SearchAddress", "OriginalAddress1", "AddressDescription", "Address"]
                    table_indexes = {}

                    for col in address_cols:
                        try:
                            cursor.execute("""
                                SELECT i.name AS index_name, i.type_desc, i.is_unique
                                FROM sys.indexes i
                                INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                                INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                                WHERE i.object_id = OBJECT_ID(?)
                                AND c.name = ?
                                AND i.type_desc != 'HEAP'
                            """, (table, col))

                            indexes = cursor.fetchall()
                            if indexes:
                                table_indexes[col] = [
                                    {"name": idx[0], "type": idx[1], "is_unique": bool(idx[2])} for idx in indexes
                                ]

                        except Exception as e:
                            table_indexes[col] = {"error": str(e)}

                    if table_indexes:
                        db_info["indexes"][table] = table_indexes

                except Exception as e:
                    db_info["tables"][table] = {"error": str(e)}

            return JSONResponse(db_info)

    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.get("/db-health")
def db_health():
    """Check database health, indexes, and table statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            health_info = {"database_connected": True, "tables": {}, "indexes": {}}

            # Check tables
            for table in ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]

                    # Get column info
                    cursor.execute(f"SELECT TOP 1 * FROM {table}")
                    columns = [column[0] for column in cursor.description]

                    health_info["tables"][table] = {
                        "row_count": row_count,
                        "columns": columns
                    }

                    # Check for indexes on address columns
                    address_cols = ["SearchAddress", "OriginalAddress1", "AddressDescription", "Address"]
                    for col in address_cols:
                        if col in columns:
                            try:
                                # Check if column has an index
                                cursor.execute("""
                                    SELECT i.name AS index_name, i.type_desc
                                    FROM sys.indexes i
                                    INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                                    INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                                    WHERE i.object_id = OBJECT_ID(?)
                                    AND c.name = ?
                                    AND i.type_desc != 'HEAP'
                                """, (table, col))

                                indexes = cursor.fetchall()
                                if indexes:
                                    if table not in health_info["indexes"]:
                                        health_info["indexes"][table] = {}
                                    health_info["indexes"][table][col] = [
                                        {"name": idx[0], "type": idx[1]} for idx in indexes
                                    ]
                            except Exception as e:
                                logging.debug("Error checking index for %s.%s: %s", table, col, e)

                except Exception as e:
                    health_info["tables"][table] = {"error": str(e)}

            # Test a simple query performance
            start_time = time.perf_counter()
            cursor.execute("SELECT TOP 1 * FROM dbo.permits")
            cursor.fetchone()
            query_time = time.perf_counter() - start_time
            health_info["query_performance_ms"] = query_time * 1000

            return JSONResponse(health_info)

    except Exception as e:
        return JSONResponse({
            "database_connected": False,
            "error": str(e)
        })

@app.get("/my-ip")
def get_my_ip():
    """Get the current outbound IP address of this Render service"""
    try:
        import urllib.request
        # Use a service to get the public IP
        ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
        return JSONResponse({
            "outbound_ip": ip,
            "message": "Add this IP to Azure SQL Server firewall rules"
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "message": "Could not determine IP address"
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

@app.get("/search-stream")
def search_stream(
    address: str = Query(..., min_length=3),
    city: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    max_results: int = Query(200),
    permit: Optional[str] = Query(None),
    street_number_q: Optional[str] = Query(None),
    street_name_q: Optional[str] = Query(None),
    street_type_q: Optional[str] = Query(None),
    street_dir_q: Optional[str] = Query(None),
    zip_q: Optional[str] = Query(None)
):
    """
    Streaming search endpoint that sends results incrementally as they're found.
    Uses Server-Sent Events (SSE) to stream results to frontend in real-time.
    """
    def generate():
        try:
            input_addr = (address or "").strip()
            if not input_addr:
                yield f"data: {json.dumps({'type': 'error', 'message': 'address parameter required'})}\n\n"
                return

            # Use Shovels API if configured
            if USE_SHOVELS_API:
                try:
                    logging.info("SEARCH-STREAM (Shovels API) | address='%s' dates=%s..%s max=%s", 
                                input_addr, date_from, date_to, max_results)
                    
                    # Format date range for Shovels API
                    permit_from = date_from if date_from else "1990-01-01"
                    permit_to = date_to if date_to else date.today().isoformat()
                    
                    # Call Shovels API
                    address_data, permits_list = get_permits_for_address(
                        address=input_addr,
                        permit_from=permit_from,
                        permit_to=permit_to
                    )
                    
                    if not address_data:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Address not found. Please verify the address and try again.'})}\n\n"
                        return
                    
                    if not permits_list:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'No permit records found for this address.'})}\n\n"
                        return
                    
                    # Filter by permit number if provided
                    if permit:
                        permits_list = [p for p in permits_list if p.get("number", "").upper() == permit.upper()]
                    
                    # Limit results
                    permits_list = permits_list[:max_results]
                    
                    # Stream each permit as it's processed
                    result_count = 0
                    for permit_data in permits_list:
                        record = map_shovels_response_to_record(address_data, permit_data)
                        rec_id = pick_id_from_record(record)
                        
                        record_data = {
                            "record_id": rec_id,
                            "permit_number": record.get("PermitNumber") or rec_id,
                            "address": record.get("SearchAddress") or record.get("OriginalAddress1") or "Address not available",
                            "city": record.get("OriginalCity") or record.get("City") or "",
                            "zip": record.get("OriginalZip") or record.get("ZipCode") or "",
                            "work_description": record.get("WorkDescription") or "",
                            "status": record.get("StatusCurrentMapped") or record.get("StatusCurrent") or "",
                            "applied_date": record.get("AppliedDate") or record.get("ApplicationDate") or "",
                            "table": "shovels_api"
                        }
                        
                        result_count += 1
                        yield f"data: {json.dumps({'type': 'record', 'data': record_data, 'count': result_count})}\n\n"
                    
                    # Send completion message
                    yield f"data: {json.dumps({'type': 'complete', 'total': result_count})}\n\n"
                    return
                    
                except ShovelsAPIError as e:
                    error_msg = str(e)
                    if "Invalid API key" in error_msg:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'System error. Please contact support.'})}\n\n"
                    elif "Rate limit" in error_msg:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Service temporarily unavailable. Please try again.'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Service temporarily unavailable. Please try again.'})}\n\n"
                    return
                except Exception as e:
                    logging.exception("Shovels API streaming error: %s", e)
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Search error: {str(e)}'})}\n\n"
                    return
            
            # Fallback to SQL database if Shovels API not configured
            # Create local variables from function parameters
            local_street_number_q = street_number_q
            local_street_name_q = street_name_q
            local_street_type_q = street_type_q
            local_street_dir_q = street_dir_q
            local_zip_q = zip_q

            with get_db_connection() as conn:
                tables = get_table_name(city)
                cursor = conn.cursor()
                all_results = []
                result_count = 0

                # Extract address components if not provided
                if not local_street_number_q and not local_street_name_q and not local_zip_q:
                    street_num, route_text, zip_code = extract_address_components(input_addr)
                    if street_num:
                        local_street_number_q = street_num
                    if route_text:
                        route_parts = [p for p in route_text.split() if p]
                        if len(route_parts) >= 1:
                            first_part = route_parts[0].lower()
                            direction_map = {
                                'n': 'N', 's': 'S', 'e': 'E', 'w': 'W',
                                'ne': 'NE', 'nw': 'NW', 'se': 'SE', 'sw': 'SW',
                                'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W'
                            }
                            if first_part in direction_map:
                                local_street_dir_q = direction_map[first_part]
                                if len(route_parts) >= 2:
                                    local_street_name_q = route_parts[1]
                                if len(route_parts) >= 3:
                                    local_street_type_q = route_parts[2]
                            else:
                                local_street_name_q = route_parts[0]
                                if len(route_parts) >= 2:
                                    second_part = route_parts[1].lower()
                                    if second_part in direction_map:
                                        local_street_dir_q = direction_map[second_part]
                                        if len(route_parts) >= 3:
                                            local_street_type_q = route_parts[2]
                                    else:
                                        local_street_type_q = route_parts[1]
                    if zip_code:
                        local_zip_q = zip_code

                # Query each table
                for table in tables:
                    try:
                        cursor.execute(f"SELECT TOP 1 * FROM {table}")
                        columns = [column[0] for column in cursor.description]
                        address_cols = []
                        for col in ["SearchAddress", "OriginalAddress1", "AddressDescription", "Address", "OriginalAddress", "StreetAddress", "PropertyAddress"]:
                            if col in columns:
                                address_cols.append(col)

                        if not address_cols:
                            continue

                        # Build query (same logic as regular search)
                        where_parts = []
                        params = []

                        if local_street_number_q and local_street_name_q:
                            address_patterns = [f"{local_street_number_q} {local_street_name_q}"]
                            if local_street_dir_q:
                                address_patterns.append(f"{local_street_number_q} {local_street_dir_q} {local_street_name_q}")
                            if local_street_type_q:
                                address_patterns.append(f"{local_street_number_q} {local_street_name_q} {local_street_type_q}")
                                if local_street_dir_q:
                                    address_patterns.append(f"{local_street_number_q} {local_street_dir_q} {local_street_name_q} {local_street_type_q}")
                            if not local_street_dir_q:
                                for direction in ['N', 'S', 'E', 'W']:
                                    address_patterns.append(f"{local_street_number_q} {direction} {local_street_name_q}")

                            pattern_conditions = []
                            for pattern in address_patterns[:6]:
                                if "SearchAddress" in address_cols:
                                    pattern_conditions.append("SearchAddress LIKE ?")
                                    params.append(f"{pattern}%")
                                for addr_col in address_cols:
                                    if addr_col != "SearchAddress":
                                        pattern_conditions.append(f"{addr_col} LIKE ?")
                                        params.append(f"{pattern}%")

                            if pattern_conditions:
                                where_parts.append(f"({' OR '.join(pattern_conditions)})")

                        if local_zip_q:
                            zip_conditions = []
                            if "OriginalZip" in columns:
                                zip_conditions.append("OriginalZip = ?")
                                params.append(local_zip_q)
                            if "ZipCode" in columns:
                                zip_conditions.append("ZipCode = ?")
                                params.append(local_zip_q)
                            if "SearchAddress" in address_cols:
                                zip_conditions.append("SearchAddress LIKE ?")
                                params.append(f"%{local_zip_q}")
                            if zip_conditions:
                                where_parts.append(f"({' OR '.join(zip_conditions)})")

                        if permit:
                            permit_cols = ["PermitNumber", "PermitNum", "Permit_Number", "Permit"]
                            for col in permit_cols:
                                if col in columns:
                                    where_parts.append(f"AND {col} = ?")
                                    params.append(permit.strip())
                                    break

                        if date_from:
                            for col in ["AppliedDate", "ApplicationDate", "DateApplied", "Date"]:
                                if col in columns:
                                    where_parts.append(f"AND {col} >= ?")
                                    params.append(date_from)
                                    break

                        if date_to:
                            for col in ["AppliedDate", "ApplicationDate", "DateApplied", "Date"]:
                                if col in columns:
                                    where_parts.append(f"AND {col} <= ?")
                                    params.append(date_to)
                                    break

                        if where_parts:
                            query = f"SELECT TOP {max_results} * FROM {table} WHERE {' AND '.join(where_parts)}"
                            if "SearchAddress" in address_cols:
                                query += " OPTION (RECOMPILE, FORCE ORDER)"

                            cursor.execute(query, params)
                            columns = [column[0] for column in cursor.description]

                            # Stream results as they're found
                            for row in cursor.fetchall():
                                rec = {col: (str(val) if val is not None else "") for col, val in zip(columns, row)}
                                rec_id = pick_id_from_record(rec)
                                
                                # Prepare record for display
                                record_data = {
                                    "record_id": rec_id,
                                    "permit_number": rec.get("PermitNumber") or rec.get("PermitNum") or rec_id,
                                    "address": rec.get("SearchAddress") or rec.get("OriginalAddress1") or rec.get("AddressDescription") or "Address not available",
                                    "city": rec.get("OriginalCity") or rec.get("City") or "",
                                    "zip": rec.get("OriginalZip") or rec.get("ZipCode") or "",
                                    "work_description": rec.get("WorkDescription") or rec.get("ProjectDescription") or rec.get("Description") or "",
                                    "status": rec.get("StatusCurrentMapped") or rec.get("CurrentStatus") or "",
                                    "applied_date": rec.get("AppliedDate") or rec.get("ApplicationDate") or "",
                                    "table": table
                                }
                                
                                result_count += 1
                                all_results.append(rec)
                                
                                # Send record immediately via SSE
                                yield f"data: {json.dumps({'type': 'record', 'data': record_data, 'count': result_count})}\n\n"

                    except Exception as e:
                        logging.exception("Error querying table %s: %s", table, e)
                        continue

                # Send completion message
                yield f"data: {json.dumps({'type': 'complete', 'total': result_count})}\n\n"

        except Exception as e:
            logging.exception("Streaming search error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/search")
def search(
    address: str = Query(..., min_length=3, description="Main search address — REQUIRED (from Google Autocomplete)"),
    city: Optional[str] = Query(None, description="Optional city"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    max_results: int = Query(200),
    permit: Optional[str] = Query(None, description="Optional permit number to match after address"),
    street_number_q: Optional[str] = Query(None, description="Optional street number from Google selection"),
    street_name_q: Optional[str] = Query(None, description="Optional street name from Google selection"),
    street_type_q: Optional[str] = Query(None, description="Optional street type/suffix from Google selection"),
    street_dir_q: Optional[str] = Query(None, description="Optional street direction (N/E/S/W) from Google selection"),
    zip_q: Optional[str] = Query(None, description="Optional zip code from Google selection")
):
    """
    Search records by address using STRICT matching criteria.
    
    Search Logic (applied in order):
    1. Address Match: If street_number AND street_name are provided, BOTH must appear together 
       in the same address field (e.g., "506 Lincoln" not "506" in one field and "Lincoln" in another)
    2. ZIP Filter: If ZIP is provided, it must match exactly (prevents wrong ZIP matches)
    3. Permit Filter: If permit number is provided, it must match exactly
    4. Date Filter: If date range is provided, records must fall within that range
    
    This ensures accurate results when:
    - User selects from Google autocomplete (components auto-filled) ✅
    - User manually types address in main field only (components extracted) ✅
    """
    start_t = time.perf_counter()
    results: List[dict] = []
    
    # Validate address parameter
    input_addr = (address or "").strip()
    if not input_addr:
        raise HTTPException(status_code=400, detail="address parameter required")
    
    # Use Shovels API if configured, otherwise fallback to SQL
    if USE_SHOVELS_API:
        try:
            logging.info("SEARCH (Shovels API) | address='%s' dates=%s..%s max=%s", 
                        input_addr, date_from, date_to, max_results)
            
            # Format date range for Shovels API
            permit_from = date_from if date_from else "1990-01-01"
            permit_to = date_to if date_to else date.today().isoformat()
            
            # Call Shovels API
            address_data, permits_list = get_permits_for_address(
                address=input_addr,
                permit_from=permit_from,
                permit_to=permit_to
            )
            
            if not address_data:
                logging.info("Shovels API: Address not found for '%s'", input_addr[:100])
                return JSONResponse({
                    "results": [],
                    "message": "Address not found. Please verify the address and try again.",
                    "duration_ms": int((time.perf_counter() - start_t) * 1000)
                })
            
            if not permits_list:
                logging.info("Shovels API: No permits found for address '%s'", input_addr[:100])
                return JSONResponse({
                    "results": [],
                    "message": "No permit records found for this address.",
                    "duration_ms": int((time.perf_counter() - start_t) * 1000)
                })
            
            # Filter by permit number if provided
            if permit:
                permits_list = [p for p in permits_list if p.get("number", "").upper() == permit.upper()]
                if not permits_list:
                    logging.info("Shovels API: No permits match permit number '%s'", permit)
                    return JSONResponse({
                        "results": [],
                        "message": f"No permits found matching permit number: {permit}",
                        "duration_ms": int((time.perf_counter() - start_t) * 1000)
                    })
            
            # Limit results
            permits_list = permits_list[:max_results]
            
            # Map each permit to record format
            for permit_data in permits_list:
                record = map_shovels_response_to_record(address_data, permit_data)
                rec_id = pick_id_from_record(record)
                record["record_id"] = rec_id
                record["permit_number"] = record.get("PermitNumber") or rec_id
                record["address"] = record.get("SearchAddress") or record.get("OriginalAddress1") or "Address not available"
                record["city"] = record.get("OriginalCity") or record.get("City") or ""
                record["zip"] = record.get("OriginalZip") or record.get("ZipCode") or ""
                record["work_description"] = record.get("WorkDescription") or record.get("Description") or ""
                record["status"] = record.get("StatusCurrentMapped") or record.get("StatusCurrent") or ""
                record["applied_date"] = record.get("AppliedDate") or record.get("ApplicationDate") or ""
                record["table"] = "shovels_api"  # Mark as from Shovels API
                results.append(record)
            
            dur_ms = int((time.perf_counter() - start_t) * 1000)
            logging.info("SEARCH (Shovels API) done | results=%d duration_ms=%d", len(results), dur_ms)
            
            return JSONResponse({
                "results": results,
                "duration_ms": dur_ms,
                "total_found": len(results)
            })
            
        except ShovelsAPIError as e:
            logging.error("Shovels API error: %s", str(e))
            error_msg = str(e)
            if "Invalid API key" in error_msg:
                return JSONResponse({
                    "results": [],
                    "error": "System error. Please contact support.",
                    "duration_ms": int((time.perf_counter() - start_t) * 1000)
                }, status_code=500)
            elif "Rate limit" in error_msg:
                return JSONResponse({
                    "results": [],
                    "error": "Service temporarily unavailable. Please try again.",
                    "duration_ms": int((time.perf_counter() - start_t) * 1000)
                }, status_code=429)
            else:
                return JSONResponse({
                    "results": [],
                    "error": "Service temporarily unavailable. Please try again.",
                    "duration_ms": int((time.perf_counter() - start_t) * 1000)
                }, status_code=500)
        except Exception as e:
            logging.exception("Shovels API search error: %s", e)
            return JSONResponse({
                "results": [],
                "error": f"Search error: {str(e)}",
                "duration_ms": int((time.perf_counter() - start_t) * 1000)
            }, status_code=500)
    
    # Fallback to SQL database if Shovels API not configured
    all_results: List[dict] = []
    try:
        with get_db_connection() as conn:
            # Determine which table(s) to query based on city
            tables = get_table_name(city)
            logging.info("SEARCH start | address='%s' city='%s' permit='%s' dates=%s..%s max=%s tables=%s",
                         input_addr, city, permit, date_from, date_to, max_results, tables)
            
            all_results = []
            cursor = conn.cursor()
            
            # Extract address components from input if not provided as separate params
            # This handles manual address entry (when user types in main field only)
            if not street_number_q and not street_name_q and not zip_q:
                street_num, route_text, zip_code = extract_address_components(input_addr)
                
                if street_num:
                    street_number_q = street_num
                
                if route_text:
                    # Parse route_text more carefully: "n lincoln ave" -> direction="N", name="lincoln", type="ave"
                    route_parts = [p for p in route_text.split() if p]  # Remove empty strings
                    
                    if len(route_parts) >= 1:
                        # Check if first part is a direction
                        first_part = route_parts[0].lower()
                        direction_map = {
                            'n': 'N', 's': 'S', 'e': 'E', 'w': 'W',
                            'ne': 'NE', 'nw': 'NW', 'se': 'SE', 'sw': 'SW',
                            'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W'
                        }
                        
                        if first_part in direction_map:
                            # First part is direction: "n lincoln ave"
                            street_dir_q = direction_map[first_part]
                            if len(route_parts) >= 2:
                                street_name_q = route_parts[1]
                            if len(route_parts) >= 3:
                                street_type_q = route_parts[2]
                        else:
                            # First part is street name: "lincoln ave"
                            street_name_q = route_parts[0]
                            if len(route_parts) >= 2:
                                # Check if second part is direction or type
                                second_part = route_parts[1].lower()
                                if second_part in direction_map:
                                    street_dir_q = direction_map[second_part]
                                    if len(route_parts) >= 3:
                                        street_type_q = route_parts[2]
                                else:
                                    street_type_q = route_parts[1]

                if zip_code:
                    zip_q = zip_code

            # Log extracted components for debugging
            logging.info("Extracted components: street_number='%s', street_name='%s', street_type='%s', street_dir='%s', zip='%s'",
                        street_number_q, street_name_q, street_type_q, street_dir_q, zip_q)

            # Validation: require at least some search criteria to prevent overly broad searches
            has_address_criteria = street_number_q or street_name_q
            has_zip_criteria = zip_q is not None
            has_permit_criteria = permit is not None
            has_date_criteria = date_from or date_to

            if not (has_address_criteria or has_zip_criteria or has_permit_criteria):
                logging.warning("Search rejected: no meaningful search criteria provided")
                return JSONResponse({
                    "results": [],
                    "message": "Please provide at least street name/number, postal code, or permit number for search.",
                    "duration_ms": 0
                })

            logging.info("Search criteria: address=%s, zip=%s, permit=%s, date_range=%s-%s",
                        has_address_criteria, has_zip_criteria, has_permit_criteria, date_from, date_to)
            
            # Query each table
            for table in tables:
                try:
                    # Get table columns first to check what address fields exist
                    cursor.execute(f"SELECT TOP 1 * FROM {table}")
                    columns = [column[0] for column in cursor.description]

                    # Find address-related columns
                    address_cols = []
                    for col in ["SearchAddress", "OriginalAddress1", "AddressDescription", "Address", "OriginalAddress", "StreetAddress", "PropertyAddress"]:
                        if col in columns:
                            address_cols.append(col)

                    if not address_cols:
                        logging.warning("No address columns found in table %s", table)
                        continue

                    # STRICT SEARCH CONDITIONS - STEP BY STEP FILTERING
                    where_parts = []
                    params = []

                    # Step 1: Address matching - STRICT: must contain both street number AND street name in same field
                    if street_number_q and street_name_q:
                        logging.info("STRICT search: street_number='%s' AND street_name='%s' must be in same address field",
                                    street_number_q, street_name_q)

                        # Build patterns that require BOTH components together (not separate)
                        address_patterns = []

                        # Primary pattern: "506 Lincoln" (both together)
                        base_pattern = f"{street_number_q} {street_name_q}"
                        address_patterns.append(base_pattern)

                        # With direction if provided: "506 N Lincoln"
                        if street_dir_q:
                            dir_pattern = f"{street_number_q} {street_dir_q} {street_name_q}"
                            address_patterns.append(dir_pattern)

                        # With street type if provided: "506 Lincoln Ave"
                        if street_type_q:
                            type_pattern = f"{street_number_q} {street_name_q} {street_type_q}"
                            address_patterns.append(type_pattern)
                            
                            # With both direction and type: "506 N Lincoln Ave"
                            if street_dir_q:
                                full_pattern = f"{street_number_q} {street_dir_q} {street_name_q} {street_type_q}"
                                address_patterns.append(full_pattern)

                        # Try common directions if no direction specified (but limit to avoid too many patterns)
                        if not street_dir_q:
                            # Only try N, S, E, W (not NE, NW, etc. to keep it focused)
                            for direction in ['N', 'S', 'E', 'W']:
                                dir_pattern = f"{street_number_q} {direction} {street_name_q}"
                                address_patterns.append(dir_pattern)
                                if street_type_q:
                                    dir_type_pattern = f"{street_number_q} {direction} {street_name_q} {street_type_q}"
                                    address_patterns.append(dir_type_pattern)

                        # Limit patterns to avoid performance issues
                        address_patterns = address_patterns[:6]

                        # Create condition: ANY address column must contain ANY of these complete patterns
                        # OPTIMIZED: Use index-friendly patterns (no leading wildcard when possible)
                        # Prefer SearchAddress column first (likely has index)
                        pattern_conditions = []
                        for pattern in address_patterns:
                            # Try index-friendly pattern first: "506 Lincoln%" (uses index)
                            # If SearchAddress exists, prioritize it
                            if "SearchAddress" in address_cols:
                                pattern_conditions.append("SearchAddress LIKE ?")
                                params.append(f"{pattern}%")  # No leading % - uses index!
                            
                            # For other columns, use both patterns for flexibility
                            for addr_col in address_cols:
                                if addr_col != "SearchAddress":  # Already handled above
                                    pattern_conditions.append(f"{addr_col} LIKE ?")
                                    params.append(f"{pattern}%")  # Index-friendly: no leading %
                                    # Also try with leading % for flexibility (but slower)
                                    pattern_conditions.append(f"{addr_col} LIKE ?")
                                    params.append(f"%{pattern}%")

                        if pattern_conditions:
                            where_parts.append(f"({' OR '.join(pattern_conditions)})")
                        else:
                            # Should not happen, but if it does, log warning
                            logging.warning("No address patterns generated for street_number='%s' street_name='%s'", 
                                          street_number_q, street_name_q)

                    elif street_number_q:
                        # Only street number - OPTIMIZED: use index-friendly patterns
                        # e.g., "506" should not match "2506"
                        logging.info("Searching for street_number='%s' only (index-optimized)", street_number_q)
                        pattern_conditions = []
                        
                        # Prioritize SearchAddress with index-friendly pattern
                        if "SearchAddress" in address_cols:
                            pattern_conditions.append("SearchAddress LIKE ?")
                            params.append(f"{street_number_q} %")  # At start - uses index!
                        
                        # Other columns with flexible matching
                        for addr_col in address_cols:
                            if addr_col != "SearchAddress":
                                pattern_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"{street_number_q} %")  # Index-friendly
                                pattern_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"% {street_number_q}%")  # Fallback
                        
                        if pattern_conditions:
                            where_parts.append(f"({' OR '.join(pattern_conditions)})")

                    elif street_name_q:
                        # Only street name - OPTIMIZED: use index-friendly patterns
                        logging.info("Searching for street_name='%s' only (index-optimized)", street_name_q)
                        pattern_conditions = []
                        
                        # Prioritize SearchAddress with index-friendly pattern
                        if "SearchAddress" in address_cols:
                            pattern_conditions.append("SearchAddress LIKE ?")
                            params.append(f"{street_name_q}%")  # No leading % - uses index!
                        
                        # Other columns
                        for addr_col in address_cols:
                            if addr_col != "SearchAddress":
                                pattern_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"{street_name_q}%")  # Index-friendly
                                pattern_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"%{street_name_q}%")  # Fallback
                        
                        if pattern_conditions:
                            where_parts.append(f"({' OR '.join(pattern_conditions)})")

                    else:
                        # No components - fallback to full address (optimized)
                        logging.info("No address components - using optimized full address search")
                        normalized_addr = normalize_text(input_addr)
                        pattern_conditions = []
                        
                        # Try to extract first word for index-friendly search
                        first_word = normalized_addr.split()[0] if normalized_addr else ""
                        
                        # Prioritize SearchAddress with index-friendly pattern
                        if "SearchAddress" in address_cols and first_word:
                            pattern_conditions.append("SearchAddress LIKE ?")
                            params.append(f"{first_word}%")  # Index-friendly
                        
                        # Full normalized address for other columns
                        for addr_col in address_cols:
                            if addr_col != "SearchAddress":
                                if first_word:
                                    pattern_conditions.append(f"{addr_col} LIKE ?")
                                    params.append(f"{first_word}%")  # Index-friendly
                                pattern_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"%{normalized_addr}%")  # Fallback
                        
                        if pattern_conditions:
                            where_parts.append(f"({' OR '.join(pattern_conditions)})")

                    # Step 2: Postal code filter (STRICT match if provided)
                    if zip_q:
                        logging.info("STRICT postal code filter: '%s'", zip_q)
                        zip_conditions = []
                        
                        # First priority: Exact match on dedicated ZIP columns
                        if "OriginalZip" in columns:
                            zip_conditions.append("OriginalZip = ?")
                            params.append(zip_q)
                        if "ZipCode" in columns:
                            zip_conditions.append("ZipCode = ?")
                            params.append(zip_q)
                        if "ZIP" in columns:
                            zip_conditions.append("ZIP = ?")
                            params.append(zip_q)
                        if "Zip" in columns:
                            zip_conditions.append("Zip = ?")
                            params.append(zip_q)

                        # Second priority: ZIP must appear in address fields (but still strict)
                        # Use word boundary to avoid partial matches like "33609" matching "336091"
                        # Also search in address fields for zip (optimized)
                        # Prioritize SearchAddress with index-friendly pattern
                        if "SearchAddress" in address_cols:
                            zip_conditions.append("SearchAddress LIKE ?")
                            params.append(f"%{zip_q}")  # ZIP at end - index-friendly
                        
                        # Other columns
                        for addr_col in address_cols:
                            if addr_col != "SearchAddress":
                                zip_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"% {zip_q}%")  # ZIP with space before
                                zip_conditions.append(f"{addr_col} LIKE ?")
                                params.append(f"%{zip_q}")    # ZIP at end

                        if zip_conditions:
                            where_parts.append(f"({' OR '.join(zip_conditions)})")
                        else:
                            logging.warning("No ZIP columns found for strict matching")

                    logging.debug("Address/Zip conditions: %d parts with %d params", len(where_parts), len(params))

                    # Step 3: Permit number filter (strict match if provided)
                    if permit:
                        logging.info("Applying permit number filter: '%s'", permit)
                        permit_cols = ["PermitNumber", "PermitNum", "Permit_Number", "Permit"]
                        permit_col = None
                        for col in permit_cols:
                            if col in columns:
                                permit_col = col
                                break
                        if permit_col:
                            where_parts.append(f"AND {permit_col} = ?")
                            params.append(permit.strip())

                    # Step 4: Date range filter (if provided)
                    if date_from or date_to:
                        date_col = None
                        for col in ["AppliedDate", "ApplicationDate", "DateApplied", "Date"]:
                            if col in columns:
                                date_col = col
                                break

                        if date_from and date_col:
                            logging.info("Applying date_from filter: '%s'", date_from)
                            where_parts.append(f"AND {date_col} >= ?")
                            params.append(date_from)

                        if date_to and date_col:
                            logging.info("Applying date_to filter: '%s'", date_to)
                            where_parts.append(f"AND {date_col} <= ?")
                            params.append(date_to)

                    # Build and execute query with performance optimizations
                    if where_parts:
                        query = f"SELECT TOP {max_results} * FROM {table} WHERE {' AND '.join(where_parts)}"
                        
                        # Add query hints to force index usage on SearchAddress if available
                        if "SearchAddress" in address_cols:
                            # Use WITH (INDEX) hint to force index usage
                            # This helps with cold start and ensures index is used
                            query += " OPTION (RECOMPILE, FORCE ORDER)"
                    else:
                        # Fallback if no conditions (shouldn't happen)
                        query = f"SELECT TOP {max_results} * FROM {table}"

                    logging.info("Final query: %s", query[:300] + "..." if len(query) > 300 else query)
                    logging.debug("Query params: %s", params)

                    logging.info("Executing query on %s: %s", table, query[:200] + "..." if len(query) > 200 else query)
                    logging.debug("Query params: %s", params)

                    query_start = time.perf_counter()
                    cursor.execute(query, params)
                    query_time = time.perf_counter() - query_start

                    # Fetch results
                    columns = [column[0] for column in cursor.description]
                    table_results = []
                    for row in cursor.fetchall():
                        rec = {col: (str(val) if val is not None else "") for col, val in zip(columns, row)}
                        table_results.append(rec)
                        all_results.append(rec)

                    # Log sample matches for debugging
                    if table_results:
                        sample_addr = table_results[0].get("SearchAddress") or table_results[0].get("OriginalAddress1") or "N/A"
                        sample_zip = table_results[0].get("OriginalZip") or table_results[0].get("ZipCode") or "N/A"
                        logging.info("Found %d results from table %s | Sample: address='%s', zip='%s'", 
                                    len(table_results), table, sample_addr[:50], sample_zip)
                    else:
                        logging.info("Found 0 results from table %s", table)
                    
                except Exception as e:
                    logging.exception("Error querying table %s: %s", table, e)
                    continue
            
            # Return results WITHOUT generating PDFs (PDF generation happens on-demand)
            # Just prepare basic record data for display
            logging.info("Processing %d records from all_results", len(all_results))
            for rec in all_results[:max_results]:
                rec_id = pick_id_from_record(rec)
                # Add basic display fields without generating PDF
                rec["record_id"] = rec_id
                rec["permit_number"] = rec.get("PermitNumber") or rec.get("PermitNum") or rec_id
                rec["address"] = rec.get("SearchAddress") or rec.get("OriginalAddress1") or rec.get("AddressDescription") or "Address not available"
                rec["city"] = rec.get("OriginalCity") or rec.get("City") or ""
                rec["zip"] = rec.get("OriginalZip") or rec.get("ZipCode") or ""
                rec["work_description"] = rec.get("WorkDescription") or rec.get("ProjectDescription") or rec.get("Description") or ""
                rec["status"] = rec.get("StatusCurrentMapped") or rec.get("CurrentStatus") or ""
                rec["applied_date"] = rec.get("AppliedDate") or rec.get("ApplicationDate") or ""
                results.append(rec)
            
            dur_ms = int((time.perf_counter() - start_t) * 1000)
            logging.info("SEARCH done | all_results=%d | results=%d duration_ms=%d | components: street_num='%s', street_name='%s', zip='%s', permit='%s'",
                         len(all_results), len(results), dur_ms, street_number_q, street_name_q, zip_q, permit)
            
            if len(results) == 0:
                logging.warning("No results to return - all_results had %d records but results list is empty", len(all_results))
                return JSONResponse({
                    "results": [],
                    "message": "Record not found. Your search data is not in records or the provided fields do not match exactly.",
                    "duration_ms": dur_ms
                })
            
            logging.info("Returning %d results to frontend", len(results))
            return JSONResponse({"results": results, "duration_ms": dur_ms, "total_found": len(results)})
    except Exception as e:
        logging.exception("Search error: %s", e)
        if "Database connection failed" in str(e) or "timeout" in str(e).lower():
            return JSONResponse({"results": [], "error": f"Database connection failed: {str(e)}"})
        return JSONResponse({"results": [], "error": str(e)})

# ---------- NEW: record endpoint now generates PDF only if record exists ----------
@app.post("/generate-pdf")
async def generate_pdf_for_record(request: Request):
    """
    Fast PDF generation endpoint for a specific record.
    Called when user clicks on a record card.
    Should complete in <1 second.
    """
    try:
        request_data = await request.json()
        permit_id = request_data.get("record_id") or request_data.get("permit_number")
        if not permit_id:
            raise HTTPException(status_code=400, detail="record_id or permit_number required")

        # Check if full record data is provided (from Shovels API or frontend)
        if "record" in request_data and isinstance(request_data["record"], dict):
            record = request_data["record"]
            source_table = record.get("table", "unknown")
            logging.info("Using provided record data for permit_id=%s from table=%s", permit_id, source_table)
        elif request_data.get("table") == "shovels_api" and USE_SHOVELS_API:
            # Record is from Shovels API - need to re-fetch it
            # This requires address and permit number
            address = request_data.get("address", "")
            if not address:
                raise HTTPException(status_code=400, detail="Address required for Shovels API record lookup")
            
            try:
                # Search address and permits
                address_data, permits_list = get_permits_for_address(address=address)
                if not address_data or not permits_list:
                    raise HTTPException(status_code=404, detail=f"Record not found for permit: {permit_id}")
                
                # Find the specific permit
                permit_data = None
                for p in permits_list:
                    if p.get("number", "").upper() == permit_id.upper():
                        permit_data = p
                        break
                
                if not permit_data:
                    raise HTTPException(status_code=404, detail=f"Permit {permit_id} not found")
                
                # Map to record format
                record = map_shovels_response_to_record(address_data, permit_data)
                source_table = "shovels_api"
                logging.info("Fetched record from Shovels API for permit_id=%s", permit_id)
            except ShovelsAPIError as e:
                logging.error("Shovels API error in generate-pdf: %s", str(e))
                raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
        else:
            # Fallback to SQL database search
            with get_db_connection() as conn:
                cursor = conn.cursor()
                tables = ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]
                record = None
                source_table = None

            for table in tables:
                try:
                    # First, get column names to check what's available
                    cursor.execute(f"SELECT TOP 1 * FROM {table}")
                    columns = [column[0] for column in cursor.description]
                    
                    # Try all ID candidate columns
                    id_cols = ID_CANDIDATES + ["PermitNumber", "PermitNum", "Permit_Number", "Permit"]
                    for col in id_cols:
                        if col in columns:
                            try:
                                cursor.execute(f"SELECT TOP 1 * FROM {table} WHERE {col} = ?", (permit_id,))
                                row = cursor.fetchone()
                                if row:
                                    record = {c: (str(val) if val is not None else "") for c, val in zip(columns, row)}
                                    source_table = table
                                    logging.info("✅ Found record in table %s using PERMIT column %s with value %s", table, col, permit_id)
                                    logging.info("Record has %d columns. Sample fields: %s", len(columns), list(columns)[:10])
                                    break
                            except Exception as e:
                                logging.debug("Error querying column %s in table %s: %s", col, table, e)
                                continue
                    if record:
                        break
                except Exception as e:
                    logging.debug("Error searching table %s: %s", table, e)
                    continue

            if not record:
                logging.warning("Record not found for permit_id=%s in any table", permit_id)
                raise HTTPException(status_code=404, detail=f"Record not found for ID: {permit_id}")

                # Store source table in record for PDF generation context
                record["_source_table"] = source_table

        # Log record details before PDF generation
        logging.info("Generating PDF for permit_id=%s from source=%s", permit_id, record.get("_source_table", "unknown"))
        logging.info("Record has %d columns. Sample fields: %s", len(record.keys()), list(record.keys())[:10])
        
        # Log key fields
        key_fields = ["PermitAddress", "Status", "StatusDesc", "PermitType", "PermitClass", 
                     "Parcel", "ProjectName", "IssuePermitDate", "PermitNumber"]
        for field in key_fields:
            if field in record:
                val = str(record[field])[:150] if record[field] else "EMPTY"
                logging.info("  [%s] = %s", field, val)

            # Generate PDF
            pdf_start = time.perf_counter()
            pdf_path = generate_pdf_from_template(record, str(TEMPLATES_DIR / "certificate-placeholder.html"))
            pdf_time = time.perf_counter() - pdf_start

            rec_id = pick_id_from_record(record)
            token = create_token_for_permit(rec_id)
            
            logging.info("PDF generated in %.2fms for record %s", pdf_time * 1000, rec_id)

            return JSONResponse({
                "success": True,
                "view_url": f"/view/{token}",
                "download_url": f"/download/{token}.pdf",
                "generation_time_ms": int(pdf_time * 1000)
            })

    except Exception as e:
        logging.exception("PDF generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

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

# ----------------- Shovels API Response Mapping -----------------
def map_shovels_response_to_record(address_data: dict, permit_data: dict) -> dict:
    """
    Map Shovels API response to existing record format for PDF generation
    
    Args:
        address_data: Address data from Shovels API (from addresses/search)
        permit_data: Permit data from Shovels API (from permits/search)
    
    Returns:
        Record dict in existing format compatible with PDF generation
    """
    # Build property address from Shovels address fields
    # Prefer "name" field (full formatted address) if available
    full_address = address_data.get("name", "")
    if not full_address:
        # Build from components if name not available
        street_no = address_data.get("street_no", "")
        street = address_data.get("street", "")
        address_parts = []
        if street_no:
            address_parts.append(street_no)
        if street:
            address_parts.append(street)
        full_address = " ".join(address_parts).strip() if address_parts else ""
    
    city = address_data.get("city", "")
    state = address_data.get("state", "")
    zip_code = address_data.get("zip_code", "")
    
    # Map permit fields
    permit_number = permit_data.get("number", "")
    permit_type = permit_data.get("type", "")
    description = permit_data.get("description", "")
    status = permit_data.get("status", "")
    
    # Map status to existing format
    status_mapped = ""
    if status == "final":
        status_mapped = "Permit Finaled"
    elif status == "active":
        status_mapped = "Active"
    elif status == "in_review":
        status_mapped = "In Review"
    elif status == "inactive":
        status_mapped = "Inactive"
    else:
        status_mapped = status.title() if status else ""
    
    # Format dates
    file_date = permit_data.get("file_date", "")
    issue_date = permit_data.get("issue_date", "")
    final_date = permit_data.get("final_date", "")
    
    # Format job value (divide by 100 for dollars)
    job_value = permit_data.get("job_value")
    if job_value:
        try:
            job_value = float(job_value) / 100.0
        except (ValueError, TypeError):
            job_value = ""
    else:
        job_value = ""
    
    # Build record in existing format
    record = {
        # Permit fields
        "PermitNumber": permit_number,
        "PermitNum": permit_number,
        "PermitType": permit_type,
        "StatusDesc": permit_type,  # For Orlando compatibility
        "WorkDescription": description,
        "ProjectDescription": description,
        "Description": description,
        "Status": status,
        "StatusCurrent": status,
        "StatusCurrentMapped": status_mapped,
        "ApplicationStatus": status,
        
        # Address fields
        "PermitAddress": full_address,
        "SearchAddress": full_address,
        "OriginalAddress1": full_address,
        "AddressDescription": full_address,
        "Address": full_address,
        "PropertyAddress": full_address,
        
        # Location fields
        "OriginalCity": city,
        "City": city,
        "PropertyCity": city,
        "OriginalState": state,
        "State": state,
        "OriginalZip": zip_code,
        "ZipCode": zip_code,
        "ZIP": zip_code,
        "Zip": zip_code,
        
        # Date fields
        "AppliedDate": file_date,
        "ApplicationDate": file_date,
        "IssuePermitDate": issue_date,  # For Orlando compatibility
        "IssueDate": issue_date,
        "CompletedDate": final_date,
        "FinalDate": final_date,
        "ExpiresDate": "",
        "LastUpdated": "",
        "StatusDate": "",
        
        # Permit class/type
        "PermitClass": permit_type,
        "PermitClassMapped": permit_type,
        "PermitClassification": permit_type,
        
        # Other fields
        "Parcel": address_data.get("parcel_id", ""),
        "ParcelNumber": address_data.get("parcel_id", ""),
        "PIN": address_data.get("parcel_id", ""),
        "ProjectName": description,  # For Orlando compatibility
        "Owner": address_data.get("property_legal_owner", ""),
        "Publisher": "Shovels API",
        "Link": "",
        
        # Job value
        "JobValue": str(job_value) if job_value else "",
        
        # Tags/categories
        "Tags": ", ".join(permit_data.get("tags", [])) if permit_data.get("tags") else "",
        
        # Jurisdiction
        "Jurisdiction": permit_data.get("jurisdiction", ""),
        
        # Contractor
        "ContractorID": permit_data.get("contractor_id", ""),
        
        # Property info (if available in address_data)
        "PropertyYearBuilt": str(address_data.get("property_year_built", "")) if address_data.get("property_year_built") else "",
        "PropertyBuildingArea": str(address_data.get("property_building_area", "")) if address_data.get("property_building_area") else "",
        "PropertyLotSize": str(address_data.get("property_lot_size", "")) if address_data.get("property_lot_size") else "",
        "PropertyType": address_data.get("property_type_detail", ""),
        "PropertyStories": str(address_data.get("property_story_count", "")) if address_data.get("property_story_count") else "",
        "PropertyAssessedValue": str(address_data.get("property_assess_market_value", "")) if address_data.get("property_assess_market_value") else "",
        
        # Coordinates
        "Latitude": str(address_data.get("lat", "")) if address_data.get("lat") else "",
        "Longitude": str(address_data.get("long", "")) if address_data.get("long") else "",
    }
    
    return record


# ----------------- PDF generation, view and download endpoints -----------------
def get_field_value(record: dict, *field_names: str) -> str:
    """Try multiple field name variations and return first non-empty value"""
    for field_name in field_names:
        value = record.get(field_name, "")
        if value and str(value).strip():
            return str(value).strip()
    return ""

def generate_certificate_number() -> str:
    """
    Generate a random 7-digit certificate number
    """
    return str(random.randint(1000000, 9999999))

def parse_address_components(address: str) -> Tuple[str, str, str]:
    """
    Extract city, state, and zip code from address string like '1708 LAKESIDE DR ORLANDO FL 32803'
    Returns (city, state, zip_code)
    """
    if not address:
        return "", "", ""
    
    # Common state abbreviations
    state_pattern = r'\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b'
    zip_pattern = r'\b(\d{5})(?:-\d{4})?\b'
    
    city = ""
    state = ""
    zip_code = ""
    
    # Find state and zip positions
    state_match = re.search(state_pattern, address.upper())
    zip_match = re.search(zip_pattern, address)
    
    if state_match:
        state = state_match.group(1)
        state_pos = state_match.start()
        before_state = address[:state_pos].strip()
        
        if zip_match:
            zip_code = zip_match.group(1)
            
            # Extract city: text between street address and state
            # Pattern: "STREET ADDRESS CITY STATE ZIP"
            # For "1708 LAKESIDE DR ORLANDO FL 32803", city is "ORLANDO"
            if before_state:
                parts = before_state.split()
                if len(parts) > 0:
                    # Common street type suffixes
                    street_types = {'DR', 'ST', 'AVE', 'AVENUE', 'BLVD', 'BOULEVARD', 'RD', 'ROAD', 
                                  'LN', 'LANE', 'CT', 'COURT', 'PL', 'PLACE', 'WAY', 'CIR', 'CIRCLE',
                                  'PKWY', 'PARKWAY', 'TER', 'TERRACE', 'TRL', 'TRAIL', 'HWY', 'HIGHWAY'}
                    
                    # Find the last street type to identify where street ends
                    last_street_type_idx = -1
                    for i, part in enumerate(parts):
                        if part.upper() in street_types:
                            last_street_type_idx = i
                    
                    # City is everything after the last street type
                    if last_street_type_idx >= 0 and last_street_type_idx < len(parts) - 1:
                        city_parts = parts[last_street_type_idx + 1:]
                        city = " ".join(city_parts).strip()
                    elif len(parts) > 1:
                        # No street type found, but multiple words - assume last word before state is city
                        # This handles cases like "123 MAIN STREET ORLANDO FL"
                        city = parts[-1]
                    else:
                        # Single word before state - likely city
                        city = parts[0]
        
        # Fallback: if we have state but no zip, still try to extract city
        if not city and before_state:
            parts = before_state.split()
            if parts:
                # Take last word as city
                city = parts[-1]
    
    # If we found zip but no state, still extract zip
    if not zip_code and zip_match:
        zip_code = zip_match.group(1)
    
    return city, state, zip_code


def generate_pdf_from_template(record: dict, template_path: str) -> str:
    tmpdir = tempfile.gettempdir()
    permit_id = pick_id_from_record(record)
    pdf_file = tmp_pdf_path_for_id(permit_id)
    try:
        logging.info("Starting PDF generation for record %s", permit_id)

        # Log ALL available columns for debugging
        all_cols = list(record.keys())
        logging.info("Record has %d columns. All columns: %s", len(all_cols), all_cols)
        
        # Log sample values for key Orlando columns
        orlando_key_cols = ["PermitAddress", "Status", "StatusDesc", "PermitType", "PermitClass", 
                           "Parcel", "ProjectName", "IssuePermitDate", "Owner"]
        for col in orlando_key_cols:
            if col in record:
                val = str(record[col])[:100] if record[col] else "EMPTY"
                logging.info("  %s = %s", col, val)

        # Address fields - try multiple variations including Orlando-specific
        addr1 = get_field_value(record, "OriginalAddress1", "OriginalAddress", "PermitAddress", 
                               "Address", "StreetAddress", "PropertyAddress", "SearchAddress")
        addr2 = get_field_value(record, "OriginalAddress2", "Address2")
        property_address = (addr1 + " " + addr2).strip() if addr2 else addr1
        
        logging.info("Address parsing: addr1='%s', addr2='%s', property_address='%s'", 
                    addr1[:100] if addr1 else "EMPTY", addr2[:50] if addr2 else "EMPTY", 
                    property_address[:100] if property_address else "EMPTY")

        # Address description - try multiple variations
        address_description = get_field_value(record, "AddressDescription", "PermitAddress", 
                                             "SearchAddress", "OriginalAddress1", "Address")

        # Parse address components once if address exists
        parsed_city = ""
        parsed_state = ""
        parsed_zip = ""
        if addr1:
            parsed_city, parsed_state, parsed_zip = parse_address_components(addr1)
            logging.info("Parsed from address '%s': city='%s', state='%s', zip='%s'", 
                        addr1[:100], parsed_city or "EMPTY", parsed_state or "EMPTY", parsed_zip or "EMPTY")

        # City - try multiple variations, then parse from address if not found
        city = get_field_value(record, "OriginalCity", "City", "PropertyCity", "PermitCity")
        if not city and parsed_city:
            city = parsed_city.title()  # Capitalize properly (e.g., "Orlando")
        if not city:
            city = "N/A"
        logging.info("Final city: '%s'", city)

        # State - try multiple variations, then parse from address if not found
        state = get_field_value(record, "OriginalState", "State", "PropertyState", "PermitState")
        if not state and parsed_state:
            state = parsed_state
        logging.info("Final state: '%s'", state or "EMPTY")

        # Zip code - try multiple variations, then parse from address if not found
        zip_code = get_field_value(record, "OriginalZip", "ZipCode", "ZIP", "Zip", "PostalCode")
        if not zip_code and parsed_zip:
            zip_code = parsed_zip
        logging.info("Final zip: '%s'", zip_code or "EMPTY")

        # Permit class - try multiple variations
        permit_class = get_field_value(record, "PermitClass", "Class", "ApplicationType", "WorkType")
        
        # Permit classification - try multiple variations
        permit_classification = get_field_value(record, "PermitClassMapped", "PermitClassification", 
                                               "Classification", "ApplicationType", "WorkType")

        # Dates - try multiple variations including Orlando-specific
        # Orlando: IssuePermitDate → Application Date
        applied_date = get_field_value(record, "IssuePermitDate", "AppliedDate", "ApplicationDate", 
                                     "DateApplied", "ApplicationDateApplied", "Date", "IssueDate")
        logging.info("Applied date: '%s' (tried IssuePermitDate, AppliedDate, etc.)", applied_date or "EMPTY")
        
        completion_date = get_field_value(record, "CompletedDate", "CompletionDate", "DateCompleted",
                                        "FinalDate", "DateFinished", "CompleteDate")
        logging.info("Completion date: '%s'", completion_date or "EMPTY")
        
        expires_date = get_field_value(record, "ExpiresDate", "ExpirationDate", "ExpiryDate",
                                      "DateExpires", "Expiration", "ExpireDate")
        logging.info("Expires date: '%s'", expires_date or "EMPTY")
        
        last_updated = get_field_value(record, "LastUpdated", "LastUpdate", "DateLastUpdated",
                                      "UpdatedDate", "ModifiedDate", "UpdateDate")
        logging.info("Last updated: '%s'", last_updated or "EMPTY")
        
        status_date = get_field_value(record, "StatusDate", "StatusDateUpdated", "DateStatus",
                                     "CurrentStatusDate", "StatusUpdateDate")
        logging.info("Status date: '%s'", status_date or "EMPTY")

        # Publisher - try multiple variations
        publisher = get_field_value(record, "Publisher", "Source", "DataSource", "Origin")

        # ensure 'other' exists so templates referencing other.* don't break
        ctx = {
            "dates": {
                "application": applied_date,
                "completion": completion_date,
                "expires": expires_date,
                "last_updated": last_updated,
                "status_date": status_date
            },
            "permit": {
                "class": permit_class,
                "classification": permit_classification,
                "number": get_field_value(record, "PermitNum", "PermitNumber", "Permit_Number", "Permit"),
                # Orlando: StatusDesc → Permit Type
                "type": get_field_value(record, "StatusDesc", "PermitType", "Type", "ApplicationType", "WorkType"),
                # Orlando: PermitType → Type Classification
                "type_classification": get_field_value(record, "PermitType", "PermitTypeMapped", "TypeClassification", "Type"),
                "id": permit_id,
                "certificate_number": generate_certificate_number(),
            },
            "property": {
                "address_description": address_description,
                "address": property_address,
                "city": city,
                "state": state,
                "zip_code": zip_code,
                # Orlando: Parcel → PIN
                "pin": get_field_value(record, "Parcel", "PIN", "ParcelNumber", "ParcelID", "ParcelNum", "PropertyID")
            },
            # Orlando: Status → Status Current
            "status_current": get_field_value(record, "Status", "StatusCurrent", "CurrentStatus", "ApplicationStatus"),
            "current_status": get_field_value(record, "StatusCurrentMapped", "CurrentStatusMapped", "StatusMapped", "ApplicationStatus", "Status"),
            "other": {
                "online_record_url": get_field_value(record, "Link", "URL", "OnlineLink", "RecordURL"),
                "publisher": publisher
            },
            # Orlando: ProjectName → Work Description
            "work_description": get_field_value(record, "ProjectName", "WorkDescription", "ProjectDescription", 
                                              "WorkDesc", "Work_Description", "WorkType", "ProjectDesc"),
            "logo_image_url": str((BASE_DIR / "Medias" / "badge.png").as_uri()) if (BASE_DIR / "Medias" / "badge.png").exists() else "",
            "map_image_url": str((BASE_DIR / "Medias" / "map.png").as_uri()) if (BASE_DIR / "Medias" / "map.png").exists() else "",
            "generated_on_date": datetime.now().strftime("%m/%d/%Y"),
            "record": record
        }
        
        # Log which fields were found for debugging
        logging.info("PDF context fields mapped:")
        logging.info("  Property: address_desc='%s', address='%s', city='%s', state='%s', zip='%s', pin='%s'",
                    address_description[:50] if address_description else "EMPTY",
                    property_address[:50] if property_address else "EMPTY",
                    city or "EMPTY", state or "EMPTY", zip_code or "EMPTY",
                    ctx["property"]["pin"] or "EMPTY")
        logging.info("  Permit: class='%s', classification='%s', type='%s', type_classification='%s'",
                    permit_class or "EMPTY", permit_classification or "EMPTY",
                    ctx["permit"]["type"] or "EMPTY", ctx["permit"]["type_classification"] or "EMPTY")
        logging.info("  Dates: application='%s', completion='%s', expires='%s', last_updated='%s', status_date='%s'",
                    applied_date or "EMPTY", completion_date or "EMPTY",
                    expires_date or "EMPTY", last_updated or "EMPTY", status_date or "EMPTY")
        logging.info("  Other: status='%s', work_desc='%s', publisher='%s'",
                    ctx["status_current"] or "EMPTY",
                    ctx["work_description"][:50] if ctx["work_description"] else "EMPTY",
                    publisher or "EMPTY")

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

# ----------------- PDF generation, view and download endpoints -----------------
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
