# address_index.py
"""
Address index for fast permit record lookups.
Can store index in Azure Blob Storage for persistence and memory efficiency.
Builds index from Azure Blob Storage CSV files and saves to Azure.
"""
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import re
import json
import pickle
from io import BytesIO

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
    """Stores location information for a record in the index (memory-optimized with __slots__)"""
    __slots__ = ('blob_name', 'row_index', 'permit_id', 'permit_num', 'date', 'city')
    
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
    
    INDEX_BLOB_NAME = "address_index.pickle"  # Name of index file in Azure
    
    def __init__(self):
        # Index: (street_number, street_name, zip_code) -> List[RecordLocation]
        self._index: Dict[Tuple[str, str, str], List[RecordLocation]] = defaultdict(list)
        self._loaded = False
        self._loading = False  # Track if index is currently being built
        self._lock = threading.Lock()
        self._build_start_time = None
        self._total_records_indexed = 0
        self._total_files_scanned = 0
        self._use_azure_storage = False  # Will be set if Azure container is available
        self._azure_container = None  # Azure container for storing index
        self._saving = False  # Flag to prevent concurrent saves
    
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
    
    def build_index_from_azure(self, src_container, progress_callback=None, max_files=None):
        """
        Build index by scanning CSV files in Azure Blob Storage.
        This is a one-time operation that may take 5-10 minutes for all files.
        
        Args:
            src_container: Azure ContainerClient for source CSVs
            progress_callback: Optional function(processed_files, total_files) called periodically
            max_files: Optional limit on number of files to index (None = index all files)
                      Useful for memory-constrained environments (e.g., index only last 10k files)
        """
        with self._lock:
            if self._loaded and len(self._index) > 0:
                logging.info("Address index already loaded in memory (%d records), skipping build", self._total_records_indexed)
                return
            # If index is marked as loaded but empty, try to reload from Azure first
            if self._loaded and len(self._index) == 0 and self._use_azure_storage and self._azure_container:
                logging.info("Index marked as loaded but empty - attempting to reload from Azure before rebuilding")
                # Release lock temporarily to allow reload
                self._lock.release()
                try:
                    if self.load_from_azure():
                        logging.info("✅ Index reloaded from Azure - no rebuild needed")
                        return
                finally:
                    self._lock.acquire()
                # If reload failed, continue with build
                logging.info("Failed to reload from Azure - will rebuild index")
            if self._loading:
                logging.info("Address index is currently being built by another thread")
                return
            self._loading = True
            self._build_start_time = datetime.now()
        
        # Check if partial index exists and resume from there (outside lock to avoid deadlock)
        resume_from = 0
        if self._use_azure_storage and self._azure_container:
            try:
                container_name = self._azure_container.container_name if hasattr(self._azure_container, 'container_name') else 'unknown'
                logging.info("🔍 Checking for existing index in Azure container: %s/%s", container_name, self.INDEX_BLOB_NAME)
                blob_client = self._azure_container.get_blob_client(self.INDEX_BLOB_NAME)
                if blob_client.exists():
                    # Get file info before loading
                    props = blob_client.get_blob_properties()
                    size_mb = props.size / (1024 * 1024)
                    logging.info("🔄 Found existing index in Azure - Size: %.1f MB - Loading to resume indexing...", size_mb)
                    if self.load_from_azure():
                        resume_from = self._total_files_scanned
                        logging.info("✅✅✅ Resuming index build from file %d (already indexed %d files, %d records) ✅✅✅", 
                                   resume_from, self._total_files_scanned, self._total_records_indexed)
                    else:
                        logging.warning("⚠️ Index file exists but failed to load - will start fresh")
                else:
                    logging.info("ℹ️ No existing index found in Azure - will start fresh indexing")
            except Exception as e:
                logging.exception("❌ Error checking for existing index: %s - will start fresh", e)
        
        try:
            logging.info("Starting address index build from Azure Blob Storage...")
            
            # Verify extraction functions are set
            if not _extract_record_address_components:
                raise RuntimeError("Extraction functions not set. Call set_extraction_functions() first.")
            
            # Check Azure storage availability and estimate space needed
            if self._use_azure_storage and self._azure_container:
                try:
                    # Test if we can write to the container (checks storage availability)
                    test_blob_name = f"_index_test_{int(time.time())}.tmp"
                    test_blob_client = self._azure_container.get_blob_client(test_blob_name)
                    test_blob_client.upload_blob(b"test", overwrite=True)
                    test_blob_client.delete_blob()
                    logging.info("✅ Azure storage write test successful - storage is accessible")
                except Exception as e:
                    logging.warning("⚠️ Azure storage write test failed: %s - indexing will continue but may fail on save", e)
            
            # List all CSV files
            all_csv_files = []
            for blob in src_container.list_blobs():
                name = getattr(blob, "name", str(blob))
                if name.lower().endswith(".csv"):
                    all_csv_files.append(name)
            
            total_files = len(all_csv_files)
            logging.info("Found %d CSV files to index", total_files)
            
            # Estimate index size (rough calculation: ~200 bytes per record on average)
            # This is conservative - actual size may be smaller due to compression in pickle
            estimated_records = total_files * 150  # Average 150 records per file
            estimated_size_mb = (estimated_records * 200) / (1024 * 1024)  # ~200 bytes per record
            estimated_size_gb = estimated_size_mb / 1024
            
            logging.info("📊 Estimated index size: ~%.1f MB (%.2f GB) for %d files", 
                        estimated_size_mb, estimated_size_gb, total_files)
            
            if estimated_size_gb > 1.0:
                logging.warning("⚠️ Large index estimated (%.2f GB) - ensure Azure storage has sufficient space", estimated_size_gb)
            elif estimated_size_mb > 500:
                logging.info("ℹ️ Index size will be approximately %.1f MB - ensure Azure storage has sufficient space", estimated_size_mb)
            else:
                logging.info("✅ Estimated index size (%.1f MB) is reasonable", estimated_size_mb)
            
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
            
            # Limit files if max_files is specified (for memory optimization)
            if max_files and max_files > 0:
                original_count = len(all_csv_files)
                all_csv_files = all_csv_files[:max_files]
                logging.info("Index build limited to %d most recent files (out of %d total) for memory optimization", 
                           len(all_csv_files), original_count)
                total_files = len(all_csv_files)  # Update total_files to reflect limit
            
            # Skip files that were already indexed (resume functionality)
            if resume_from > 0 and resume_from < len(all_csv_files):
                all_csv_files = all_csv_files[resume_from:]
                logging.info("⏩ Skipping %d already-indexed files, continuing from file %d", resume_from, resume_from + 1)
                total_files = len(all_csv_files) + resume_from  # Total includes already indexed
            
            # Import pandas here to avoid circular dependency
            import pandas as pd
            from io import BytesIO
            
            # Scan each CSV file
            processed_files = resume_from  # Start from resume point
            start_time = time.time()
            last_progress_time = start_time
            
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
                            # Note: street_name from extract_record_address_components is already normalized
                            street_number = (street_number or "").strip()
                            street_name = (street_name or "").strip() if street_name else ""
                            # Ensure street_name is normalized (extract_record_address_components should already normalize, but double-check)
                            if street_name and _normalize_text:
                                street_name = _normalize_text(street_name)
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
                    
                    # Save checkpoint at multiple intervals to prevent data loss on server restart
                    # Save at 4900 (before 5000) to avoid restart during save, and at 1000, 2000, 3000, 4000 for frequent backups
                    checkpoint_intervals = [1000, 2000, 3000, 4000, 4900, 5000]
                    if processed_files in checkpoint_intervals and self._use_azure_storage and self._azure_container and not self._saving:
                        try:
                            self._saving = True  # Prevent concurrent saves
                            # Estimate current index size
                            current_size_estimate = (self._total_records_indexed * 200) / (1024 * 1024)  # MB
                            logging.info("💾 Saving checkpoint at %d/%d files (%.1f%%) | Current index: ~%.1f MB...", 
                                       processed_files, total_files, (processed_files/total_files*100), current_size_estimate)
                            
                            # Save with timing
                            save_start = time.time()
                            self._save_to_azure()
                            save_duration = time.time() - save_start
                            
                            # Verify the file was actually saved
                            blob_client = self._azure_container.get_blob_client(self.INDEX_BLOB_NAME)
                            if blob_client.exists():
                                props = blob_client.get_blob_properties()
                                actual_size_mb = props.size / (1024 * 1024)
                                container_name = self._azure_container.container_name if hasattr(self._azure_container, 'container_name') else 'container'
                                logging.info("✅✅✅ Checkpoint saved successfully in %.1f seconds - Actual size: %.1f MB ✅✅✅", save_duration, actual_size_mb)
                                logging.info("✅ Checkpoint verification: File exists in Azure at %s/%s", container_name, self.INDEX_BLOB_NAME)
                            else:
                                logging.error("❌❌❌ CRITICAL: Checkpoint save reported success but file not found in Azure! Index may be lost on restart!")
                        except Exception as e:
                            logging.exception("❌❌❌ Failed to save checkpoint: %s - indexing will continue but progress may be lost on restart", e)
                        finally:
                            self._saving = False  # Release save lock
                    
                    # Progress logging every 100 files
                    if processed_files % 100 == 0 or processed_files == total_files:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        files_per_sec = processed_files / elapsed if elapsed > 0 else 0
                        remaining_files = total_files - processed_files
                        eta_seconds = remaining_files / files_per_sec if files_per_sec > 0 else 0
                        eta_minutes = eta_seconds / 60
                        progress_pct = (processed_files / total_files * 100) if total_files > 0 else 0
                        
                        logging.info("Index build progress: %d/%d files (%.1f%%) | %d records, %d unique addresses | ETA: %.1f minutes",
                                   processed_files, total_files, progress_pct, self._total_records_indexed, len(self._index), eta_minutes)
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
            logging.info("=" * 80)
            logging.info("✅✅✅ ADDRESS INDEX BUILD COMPLETED SUCCESSFULLY! ✅✅✅")
            logging.info("   Duration: %.1f seconds (%.1f minutes)", build_duration, build_duration / 60)
            logging.info("   Files indexed: %d / %d (100%%)", self._total_files_scanned, total_files)
            logging.info("   Records indexed: %d", self._total_records_indexed)
            logging.info("   Unique addresses: %d", len(self._index))
            logging.info("   Status: Index is now ready for fast searches!")
            logging.info("=" * 80)
            
            # Save index to Azure Storage and clear from memory to save RAM
            if self._use_azure_storage and self._azure_container:
                try:
                    logging.info("💾 Saving index to Azure Blob Storage...")
                    self._save_to_azure()
                    logging.info("✅✅✅ Index saved to Azure Blob Storage: %s ✅✅✅", self.INDEX_BLOB_NAME)
                    # Keep index in memory for fast lookups
                    # Index is saved to Azure for persistence across server restarts
                    # Memory usage is acceptable for performance benefits
                    logging.info("✅ Index kept in memory for fast lookups")
                    logging.info("=" * 80)
                    logging.info("🎉🎉🎉 INDEXING COMPLETE - ALL FUTURE SEARCHES WILL BE FAST! 🎉🎉🎉")
                    logging.info("=" * 80)
                except Exception as e:
                    logging.warning("Failed to save index to Azure: %s", e)
                    raise  # Re-raise so index stays in memory if save fails
        
        except Exception as e:
            logging.exception("Failed to build address index: %s", e)
            with self._lock:
                self._loading = False
            raise
    
    def get_indexed_files(self) -> set:
        """Get set of all blob names that have been indexed (for partial index support)"""
        with self._lock:
            indexed_files = set()
            for locations in self._index.values():
                for loc in locations:
                    indexed_files.add(loc.blob_name)
            return indexed_files
    
    def lookup(self, street_number: Optional[str], street_name: Optional[str], 
               zip_code: Optional[str], date_from: Optional[str] = None, 
               date_to: Optional[str] = None, permit: Optional[str] = None,
               city: Optional[str] = None) -> List[RecordLocation]:
        """
        Look up records by address components.
        Works with both full and partial index (returns matches from indexed files only).
        
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
        # If index is marked as loaded but empty in memory, reload from Azure
        if self._loaded and len(self._index) == 0 and self._use_azure_storage and self._azure_container:
            logging.debug("Index marked as loaded but empty in memory - reloading from Azure")
            if not self.load_from_azure():
                logging.warning("Index was marked as loaded but failed to reload from Azure - clearing loaded flag")
                with self._lock:
                    self._loaded = False
                return []
        
        # If index not loaded and not loading, try to load from Azure
        if not self._loaded and not self._loading and self._use_azure_storage and self._azure_container:
            if self.load_from_azure():
                logging.info("Index loaded from Azure on-demand for lookup")
            else:
                return []  # No index available
        
        # Normalize inputs (same as index build)
        street_number = (street_number or "").strip() if street_number else None
        street_name = (street_name or "").strip() if street_name else None
        zip_code = (zip_code or "").strip()[:5] if zip_code else None
        
        # Normalize street_name using the same function as index build
        if street_name and _normalize_text:
            street_name = _normalize_text(street_name)
        
        # Build lookup key
        index_key = (street_number or "", street_name or "", zip_code or None)
        
        # Debug logging
        logging.debug("Index lookup key: street_number='%s' street_name='%s' zip='%s'", 
                     street_number, street_name, zip_code)
        
        # Get all matches for this key
        with self._lock:
            matches = self._index.get(index_key, [])
            # Also try partial matches if exact match fails
            if not matches:
                # Try without zip code
                if zip_code:
                    partial_key = (street_number or "", street_name or "", None)
                    matches = self._index.get(partial_key, [])
                    if matches:
                        logging.debug("Index lookup: Found %d matches without zip code", len(matches))
                # Try without street number
                if not matches and street_number:
                    partial_key = ("", street_name or "", zip_code or None)
                    matches = self._index.get(partial_key, [])
                    if matches:
                        logging.debug("Index lookup: Found %d matches without street number", len(matches))
        
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
    
    def set_azure_storage(self, container_client):
        """Set Azure container for persisting index"""
        self._azure_container = container_client
        self._use_azure_storage = container_client is not None
    
    def _save_to_azure(self):
        """Save index to Azure Blob Storage as pickle file"""
        if not self._azure_container:
            return
        
        # Convert index to serializable format
        index_data = {
            "index": {},
            "total_records": self._total_records_indexed,
            "total_files": self._total_files_scanned,
            "build_time": self._build_start_time.isoformat() if self._build_start_time else None,
            "version": "1.0"
        }
        
        # Convert tuple keys to strings and RecordLocation to dicts
        for key, locations in self._index.items():
            key_str = f"{key[0]}|{key[1]}|{key[2]}"  # Convert tuple to string
            index_data["index"][key_str] = [loc.to_dict() for loc in locations]
        
        # Serialize to pickle (more efficient than JSON for large data)
        pickle_data = pickle.dumps(index_data)
        
        # Upload to Azure
        try:
            blob_client = self._azure_container.get_blob_client(self.INDEX_BLOB_NAME)
            container_name = self._azure_container.container_name if hasattr(self._azure_container, 'container_name') else 'unknown'
            size_mb = len(pickle_data) / (1024 * 1024)
            logging.info("📤 Uploading index to Azure: %s/%s (%.1f MB)...", container_name, self.INDEX_BLOB_NAME, size_mb)
            blob_client.upload_blob(pickle_data, overwrite=True)
            
            # Verify upload
            if not blob_client.exists():
                raise Exception("Upload reported success but file not found in Azure!")
            
            size_mb = len(pickle_data) / (1024 * 1024)
            size_gb = size_mb / 1024
            if size_gb >= 1.0:
                logging.info("✅ Index saved to Azure: %s/%s (%.2f GB, %d bytes)", container_name, self.INDEX_BLOB_NAME, size_gb, len(pickle_data))
            else:
                logging.info("✅ Index saved to Azure: %s/%s (%.1f MB, %d bytes)", container_name, self.INDEX_BLOB_NAME, size_mb, len(pickle_data))
        except Exception as e:
            logging.exception("❌❌❌ CRITICAL ERROR saving index to Azure: %s", e)
            raise  # Re-raise so caller knows save failed
    
    def load_from_azure(self) -> bool:
        """
        Load index from Azure Blob Storage.
        Returns True if loaded successfully, False otherwise.
        """
        if not self._azure_container:
            return False
        
        try:
            blob_client = self._azure_container.get_blob_client(self.INDEX_BLOB_NAME)
            if not blob_client.exists():
                logging.info("Index file not found in Azure: %s", self.INDEX_BLOB_NAME)
                return False
            
            # Download and deserialize
            pickle_data = blob_client.download_blob().readall()
            index_data = pickle.loads(pickle_data)
            
            # Reconstruct index
            with self._lock:
                self._index.clear()
                for key_str, locations_dict in index_data["index"].items():
                    # Convert string key back to tuple
                    parts = key_str.split("|")
                    key = (parts[0] if parts[0] else None, 
                          parts[1] if parts[1] else None, 
                          parts[2] if parts[2] else None)
                    # Reconstruct RecordLocation objects
                    self._index[key] = [RecordLocation(**loc_dict) for loc_dict in locations_dict]
                
                self._total_records_indexed = index_data.get("total_records", 0)
                self._total_files_scanned = index_data.get("total_files", 0)
                self._loaded = True
                self._loading = False
            
            logging.info("✅ Index loaded from Azure: %d records, %d unique addresses, %d files",
                       self._total_records_indexed, len(self._index), self._total_files_scanned)
            return True
        
        except Exception as e:
            logging.warning("Failed to load index from Azure: %s", e)
            return False
    
    def clear(self):
        """Clear the index (for testing or rebuilding)"""
        with self._lock:
            self._index.clear()
            self._loaded = False
            self._loading = False
            self._total_records_indexed = 0
            self._total_files_scanned = 0
            self._build_start_time = None

