"""
Shovels API Client Module
Handles all interactions with the Shovels API for permit data retrieval.
"""
import os
import requests
import logging
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime

# ----------------- Configuration -----------------
SHOVELS_API_KEY = os.getenv("SHOVELS_API_KEY", "")
SHOVELS_BASE_URL = "https://api.shovels.ai/v2"
SHOVELS_TIMEOUT = 30  # seconds

# ----------------- Logging -----------------
logger = logging.getLogger(__name__)


class ShovelsAPIError(Exception):
    """Custom exception for Shovels API errors"""
    pass


def get_shovels_headers() -> Dict[str, str]:
    """Get headers for Shovels API requests"""
    if not SHOVELS_API_KEY:
        raise ShovelsAPIError("SHOVELS_API_KEY environment variable is not set")
    return {"X-API-Key": SHOVELS_API_KEY}


def search_address(address: str) -> Optional[Dict]:
    """
    Step 1: Search for address and get geo_id
    
    Args:
        address: User's address input (e.g., "5116 Stonehurst Rd Tampa FL")
    
    Returns:
        Address data dict with geo_id, or None if not found
    
    Raises:
        ShovelsAPIError: If API call fails
    """
    if not address or not address.strip():
        raise ShovelsAPIError("Address parameter is required")
    
    try:
        url = f"{SHOVELS_BASE_URL}/addresses/search"
        headers = get_shovels_headers()
        params = {"q": address.strip()}
        
        logger.info("Shovels API: Searching address '%s'", address[:100])
        
        response = requests.get(url, headers=headers, params=params, timeout=SHOVELS_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("items") or len(data["items"]) == 0:
            logger.info("Shovels API: No address found for '%s'", address[:100])
            return None
        
        address_data = data["items"][0]
        logger.info("Shovels API: Found address with geo_id=%s", address_data.get("geo_id"))
        
        return address_data
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ShovelsAPIError("Invalid API key. Please check SHOVELS_API_KEY environment variable.")
        elif e.response.status_code == 429:
            raise ShovelsAPIError("Rate limit exceeded. Please try again later.")
        elif e.response.status_code >= 500:
            raise ShovelsAPIError("Shovels API server error. Please try again later.")
        else:
            raise ShovelsAPIError(f"Shovels API error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.Timeout:
        raise ShovelsAPIError("Shovels API request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise ShovelsAPIError(f"Network error connecting to Shovels API: {str(e)}")
    except Exception as e:
        raise ShovelsAPIError(f"Unexpected error in address search: {str(e)}")


def search_permits(geo_id: str, permit_from: Optional[str] = None, permit_to: Optional[str] = None) -> List[Dict]:
    """
    Step 2: Search for permits using geo_id with cursor-based pagination
    
    Fetches ALL permits by looping through all pages until next_cursor is null.
    
    Args:
        geo_id: Address geo_id from Step 1
        permit_from: Start date (YYYY-MM-DD). Default: 1990-01-01 (for client-side filtering)
        permit_to: End date (YYYY-MM-DD). Default: today (for client-side filtering)
    
    Returns:
        List of all permit records (all pages combined)
    
    Raises:
        ShovelsAPIError: If API call fails
    """
    if not geo_id:
        raise ShovelsAPIError("geo_id parameter is required")
    
    # Default date range: 1990-01-01 to today (used for client-side filtering after fetch)
    if not permit_from:
        permit_from = "1990-01-01"
    if not permit_to:
        permit_to = date.today().isoformat()
    
    try:
        all_permits = []
        cursor = None
        page_num = 1
        max_page_size = 500  # Maximum allowed by Shovels API
        
        logger.info("Shovels API: Searching permits for geo_id=%s (with pagination), date filter: %s to %s", 
                   geo_id, permit_from, permit_to)
        
        while True:
            # Use /addresses/{geo_id}/permits endpoint with pagination
            url = f"{SHOVELS_BASE_URL}/addresses/{geo_id}/permits"
            headers = get_shovels_headers()
            params = {"page_size": max_page_size}
            
            if cursor:
                params["cursor"] = cursor
            
            response = requests.get(url, headers=headers, params=params, timeout=SHOVELS_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            page_items = data.get("items", [])
            all_permits.extend(page_items)
            
            page_size = data.get("size", len(page_items))
            next_cursor = data.get("next_cursor")
            
            logger.info("Shovels API: Page %d - fetched %d permits (total so far: %d), next_cursor: %s", 
                       page_num, page_size, len(all_permits), "present" if next_cursor else "null")
            
            # Break if no more pages
            if not next_cursor:
                break
            
            cursor = next_cursor
            page_num += 1
        
        logger.info("Shovels API: Completed pagination - found %d total permits for geo_id=%s", 
                   len(all_permits), geo_id)
        
        # Apply date filtering client-side if dates were specified
        if permit_from != "1990-01-01" or permit_to != date.today().isoformat():
            filtered_permits = []
            permit_from_date = datetime.strptime(permit_from, "%Y-%m-%d").date()
            permit_to_date = datetime.strptime(permit_to, "%Y-%m-%d").date()
            
            for permit in all_permits:
                # Try to get permit date from various fields
                permit_date_str = (permit.get("permit_date") or 
                                 permit.get("issue_date") or 
                                 permit.get("applied_date") or 
                                 permit.get("date"))
                
                if permit_date_str:
                    permit_date = None
                    try:
                        # Try parsing different date formats
                        if isinstance(permit_date_str, str):
                            # Try ISO format first
                            try:
                                permit_date = datetime.fromisoformat(permit_date_str.replace('Z', '+00:00')).date()
                            except:
                                # Try other common formats
                                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
                                    try:
                                        permit_date = datetime.strptime(permit_date_str[:10], fmt).date()
                                        break
                                    except:
                                        continue
                        
                        # If we successfully parsed a date, check if it's within range
                        if permit_date:
                            if permit_from_date <= permit_date <= permit_to_date:
                                filtered_permits.append(permit)
                        else:
                            # Include permit if date parsing fails (better to show than hide)
                            filtered_permits.append(permit)
                    except Exception as e:
                        logger.debug("Could not parse permit date '%s': %s", permit_date_str, e)
                        # Include permit if date parsing fails (better to show than hide)
                        filtered_permits.append(permit)
                else:
                    # Include permit if no date available (better to show than hide)
                    filtered_permits.append(permit)
            
            logger.info("Shovels API: Date filter applied - %d permits match date range %s to %s", 
                       len(filtered_permits), permit_from, permit_to)
            return filtered_permits
        
        return all_permits
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ShovelsAPIError("Invalid API key. Please check SHOVELS_API_KEY environment variable.")
        elif e.response.status_code == 429:
            raise ShovelsAPIError("Rate limit exceeded. Please try again later.")
        elif e.response.status_code >= 500:
            raise ShovelsAPIError("Shovels API server error. Please try again later.")
        else:
            raise ShovelsAPIError(f"Shovels API error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.Timeout:
        raise ShovelsAPIError("Shovels API request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise ShovelsAPIError(f"Network error connecting to Shovels API: {str(e)}")
    except Exception as e:
        raise ShovelsAPIError(f"Unexpected error in permit search: {str(e)}")


def get_permits_for_address(address: str, permit_from: Optional[str] = None, 
                           permit_to: Optional[str] = None) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Complete workflow: Get address and all permits for an address
    
    Args:
        address: User's address input
        permit_from: Start date for permit search (YYYY-MM-DD). Default: 1990-01-01
        permit_to: End date for permit search (YYYY-MM-DD). Default: today
    
    Returns:
        Tuple of (address_data, permits_list)
        - address_data: Address info with geo_id, or None if address not found
        - permits_list: List of permit records, or empty list if no permits found
    
    Raises:
        ShovelsAPIError: If API calls fail
    """
    # Step 1: Search address
    address_data = search_address(address)
    
    if not address_data:
        return None, []
    
    geo_id = address_data.get("geo_id")
    if not geo_id:
        logger.warning("Shovels API: Address found but no geo_id in response")
        return address_data, []
    
    # Step 2: Search permits
    permits = search_permits(geo_id, permit_from, permit_to)
    
    return address_data, permits


