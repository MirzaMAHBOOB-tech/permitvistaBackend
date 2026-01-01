# address_index.py
"""
In-memory address index for fast permit record lookups.
Builds index from Azure Blob Storage CSV files on first request (lazy loading).
"""
import logging
import threading
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import re

# Import matching functions from api_server
# We'll import these at runtime to avoid circular imports
_extract_record_address_components = None
_normalize_text = None
_pick_id_from_record = None

def set_extraction_functions(extract_func, normalize_func, pick_id_func):
    """Set the extraction functions from api_server (called during initialization)"""
    global _extract_record_address_components, _normalize_text, _pick_id_from_record
    _extract_record_address_components = extract_func
    _normalize_text = normalize_func
    _pick_id_from_record = pick_id_func


class RecordLocation:
    """Stores location information for a record in the index"""
    def __init__(self, blob_name: str, row_index: int, permit_id: str, permit_num: str, 
                 date: str, city: str):
        self.blob_name = blob_name
        self.row_index = row_index
        self.permit_id = permit_id
        self.permit_num = permit_num
        self.date = date  # AppliedDate or file date
        self.city = city  # OriginalCity or City
    
    def to_dict(self):
        return {
            "blob_name": self.blob_name,
            "row_index": self.row_index,
            "permit_id": self.permit_id,
            "permit_num": self.permit_num,
            "date": self.date,
            "city": self.city
        }


class AddressIndex:
    """
    In-memory index mapping (street_number, street_name, zip_code) to record locations.
    Index structure: Dict[Tuple[str, str, str], List[RecordLocation]]
    """
    
    def __init__(self):
        # Index: (street_number, street_name, zip_code) -> List[RecordLocation]
        self._index: Dict[Tuple[str, str, str], List[RecordLocation]] = defaultdict(list)
        self._loaded = False
        self._loading = False  # Track if index is currently being built
        self._lock = threading.Lock()
        self._build_start_time = None
        self._total_records_indexed = 0
        self._total_files_scanned = 0
    
    def is_loaded(self) -> bool:
        """Check if index is fully loaded"""
        return self._loaded
    
    def is_loading(self) -> bool:
        """Check if index is currently being built"""
        return self._loading
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        with self._lock:
            return {
                "loaded": self._loaded,
                "loading": self._loading,
                "total_records": self._total_records_indexed,
                "total_files": self._total_files_scanned,
                "unique_addresses": len(self._index),
                "build_start_time": self._build_start_time.isoformat() if self._build_start_time else None
            }
    
    def build_index_from_azure(self, src_container, progress_callback=None):
        """
        Build index by scanning all CSV files in Azure Blob Storage.
        This is a one-time operation that may take 5-10 minutes.
        
        Args:
            src_container: Azure ContainerClient for source CSVs
            progress_callback: Optional function(processed_files, total_files) called periodically
        """
        with self._lock:
            if self._loaded:
                logging.info("Address index already loaded, skipping build")
                return
            if self._loading:
                logging.info("Address index is currently being built by another thread")
                return
            self._loading = True
            self._build_start_time = datetime.now()
        
        try:
            logging.info("Starting address index build from Azure Blob Storage...")
            
            # Verify extraction functions are set
            if not _extract_record_address_components:
                raise RuntimeError("Extraction functions not set. Call set_extraction_functions() first.")
            
            # List all CSV files
            all_csv_files = []
            for blob in src_container.list_blobs():
                name = getattr(blob, "name", str(blob))
                if name.lower().endswith(".csv"):
                    all_csv_files.append(name)
            
            total_files = len(all_csv_files)
            logging.info("Found %d CSV files to index", total_files)
            
            # Extract date from path for sorting (newest first)
            def extract_date_from_path(path):
                parts = path.split('/')
                if len(parts) >= 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        return (year, month, day)
                    except (ValueError, IndexError):
                        pass
                date_match = re.search(r'(\d{4})(\d{2})(\d{2})', path)
                if date_match:
                    return (int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
                return (0, 0, 0)
            
            all_csv_files.sort(key=extract_date_from_path, reverse=True)
            
            # Import pandas here to avoid circular dependency
            import pandas as pd
            from io import BytesIO
            
            # Scan each CSV file
            processed_files = 0
            for csv_file_name in all_csv_files:
                try:
                    # Read CSV from Azure
                    blob_client = src_container.get_blob_client(csv_file_name)
                    if not blob_client.exists():
                        continue
                    
                    data = blob_client.download_blob().readall()
                    df = pd.read_csv(BytesIO(data), dtype=str)
                    
                    # Process each row
                    for row_index, (_, row) in enumerate(df.iterrows()):
                        try:
                            rec = {k.strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                            
                            # Extract address components using same logic as search()
                            street_number, street_name, zip_code = _extract_record_address_components(rec)
                            
                            # Only index if we have at least street_number or street_name
                            if not street_number and not street_name:
                                continue
                            
                            # Normalize values (same as search logic)
                            street_number = (street_number or "").strip()
                            street_name = (street_name or "").strip() if street_name else ""
                            zip_code = (zip_code or "").strip()[:5] if zip_code else None
                            
                            # Create index key
                            index_key = (street_number, street_name, zip_code or None)
                            
                            # Get permit ID and other metadata
                            permit_id = _pick_id_from_record(rec) or ""
                            permit_num = (rec.get("PermitNum") or rec.get("PermitNumber") or "").strip()
                            applied_date = (rec.get("AppliedDate") or "").strip()
                            
                            # Extract date from file path if AppliedDate not available
                            if not applied_date:
                                file_date_tuple = extract_date_from_path(csv_file_name)
                                if file_date_tuple[0] > 0:
                                    applied_date = f"{file_date_tuple[0]:04d}-{file_date_tuple[1]:02d}-{file_date_tuple[2]:02d}"
                            
                            city = _normalize_text(rec.get("OriginalCity") or rec.get("City") or "")
                            
                            # Create record location
                            location = RecordLocation(
                                blob_name=csv_file_name,
                                row_index=row_index,
                                permit_id=permit_id,
                                permit_num=permit_num,
                                date=applied_date,
                                city=city
                            )
                            
                            # Add to index
                            with self._lock:
                                self._index[index_key].append(location)
                                self._total_records_indexed += 1
                        
                        except Exception as e:
                            logging.debug("Error processing row %d in %s: %s", row_index, csv_file_name, e)
                            continue
                    
                    processed_files += 1
                    self._total_files_scanned = processed_files
                    
                    # Progress logging every 100 files
                    if processed_files % 100 == 0 or processed_files == total_files:
                        logging.info("Index build progress: %d/%d files processed, %d records indexed, %d unique addresses",
                                   processed_files, total_files, self._total_records_indexed, len(self._index))
                        if progress_callback:
                            progress_callback(processed_files, total_files)
                
                except Exception as e:
                    logging.warning("Error processing CSV file %s: %s", csv_file_name, e)
                    continue
            
            # Mark as loaded
            with self._lock:
                self._loaded = True
                self._loading = False
            
            build_duration = (datetime.now() - self._build_start_time).total_seconds()
            logging.info("Address index build completed in %.1f seconds: %d records indexed, %d unique addresses, %d files scanned",
                       build_duration, self._total_records_indexed, len(self._index), self._total_files_scanned)
        
        except Exception as e:
            logging.exception("Failed to build address index: %s", e)
            with self._lock:
                self._loading = False
            raise
    
    def lookup(self, street_number: Optional[str], street_name: Optional[str], 
               zip_code: Optional[str], date_from: Optional[str] = None, 
               date_to: Optional[str] = None, permit: Optional[str] = None,
               city: Optional[str] = None) -> List[RecordLocation]:
        """
        Look up records by address components.
        
        Args:
            street_number: Street number (e.g., "1508")
            street_name: Normalized street name (e.g., "park ln w")
            zip_code: ZIP code (first 5 digits)
            date_from: Optional date filter (YYYY-MM-DD)
            date_to: Optional date filter (YYYY-MM-DD)
            permit: Optional permit number filter
            city: Optional city filter (normalized)
        
        Returns:
            List of RecordLocation objects matching the criteria
        """
        if not self._loaded:
            return []
        
        # Normalize inputs
        street_number = (street_number or "").strip() if street_number else None
        street_name = (street_name or "").strip() if street_name else None
        zip_code = (zip_code or "").strip()[:5] if zip_code else None
        
        # Build lookup key
        index_key = (street_number or "", street_name or "", zip_code or None)
        
        # Get all matches for this key
        with self._lock:
            matches = self._index.get(index_key, [])
        
        # Apply filters
        filtered = []
        for location in matches:
            # Date filter
            if date_from and location.date and location.date < date_from:
                continue
            if date_to and location.date and location.date > date_to:
                continue
            
            # Permit filter
            if permit:
                permit_lower = permit.strip().lower()
                loc_permit_lower = (location.permit_num or "").strip().lower()
                if not loc_permit_lower or permit_lower != loc_permit_lower:
                    continue
            
            # City filter
            if city:
                city_norm = _normalize_text(city) if _normalize_text else city.lower()
                if city_norm not in location.city:
                    continue
            
            filtered.append(location)
        
        return filtered
    
    def clear(self):
        """Clear the index (for testing or rebuilding)"""
        with self._lock:
            self._index.clear()
            self._loaded = False
            self._loading = False
            self._total_records_indexed = 0
            self._total_files_scanned = 0
            self._build_start_time = None

