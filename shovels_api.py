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
    Step 2: Search for permits using geo_id
    
    Args:
        geo_id: Address geo_id from Step 1
        permit_from: Start date (YYYY-MM-DD). Default: 1990-01-01
        permit_to: End date (YYYY-MM-DD). Default: today
    
    Returns:
        List of permit records
    
    Raises:
        ShovelsAPIError: If API call fails
    """
    if not geo_id:
        raise ShovelsAPIError("geo_id parameter is required")
    
    # Default date range: 1990-01-01 to today
    if not permit_from:
        permit_from = "1990-01-01"
    if not permit_to:
        permit_to = date.today().isoformat()
    
    try:
        url = f"{SHOVELS_BASE_URL}/permits/search"
        headers = get_shovels_headers()
        params = {
            "geo_id": geo_id,
            "permit_from": permit_from,
            "permit_to": permit_to
        }
        
        logger.info("Shovels API: Searching permits for geo_id=%s, dates=%s to %s", 
                   geo_id, permit_from, permit_to)
        
        response = requests.get(url, headers=headers, params=params, timeout=SHOVELS_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        permits = data.get("items", [])
        
        logger.info("Shovels API: Found %d permits for geo_id=%s", len(permits), geo_id)
        
        return permits
        
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




