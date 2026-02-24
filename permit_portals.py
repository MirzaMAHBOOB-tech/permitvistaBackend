"""
Florida Permit Portal URLs
Maps jurisdiction names (as returned by Shovels API) to their online permit portals.

Usage:
    from permit_portals import get_permit_portal_url
    
    url = get_permit_portal_url("MIAMI BEACH", "BCR2504620")
    # Returns: "https://www.miamibeachfl.gov/city-hall/building/online-permitting/"
"""

# Permit portal configurations
# format: "JURISDICTION_NAME": {
#     "search_url": URL for searching permits (if available),
#     "base_url": General permit portal URL (fallback),
#     "deep_link": True if search_url supports permit number lookup
# }

PERMIT_PORTALS = {
    # =========================================================================
    # MIAMI-DADE COUNTY
    # =========================================================================
    "MIAMI": {
        "search_url": "https://www.miamigov.com/My-Government/Departments/Building/Permit-Search",
        "base_url": "https://www.miamigov.com/My-Government/Departments/Building",
        "deep_link": False,
    },
    "MIAMI BEACH": {
        "search_url": "https://www.miamibeachfl.gov/city-hall/building/online-permitting/",
        "base_url": "https://www.miamibeachfl.gov/city-hall/building/",
        "deep_link": False,
    },
    "CORAL GABLES": {
        "search_url": "https://www.coralgables.com/departments/development-services/building-division/permit-search",
        "base_url": "https://www.coralgables.com/departments/development-services/building-division",
        "deep_link": False,
    },
    "HALLANDALE BEACH": {
        "search_url": "https://www.hallandalebeachfl.gov/149/Building-Services",
        "base_url": "https://www.hallandalebeachfl.gov/149/Building-Services",
        "deep_link": False,
    },
    
    # =========================================================================
    # BROWARD COUNTY
    # =========================================================================
    "FORT LAUDERDALE": {
        "search_url": "https://aca-prod.accela.com/FTLAUDERDALE/Cap/CapHome.aspx?module=Building",
        "base_url": "https://www.fortlauderdale.gov/departments/development-services/building-services",
        "deep_link": False,  # Accela - can potentially deep link with capID
    },
    "POMPANO BEACH": {
        "search_url": "https://pompanobeachfl.gov/pages/development_services/building_division",
        "base_url": "https://pompanobeachfl.gov/pages/development_services/building_division",
        "deep_link": False,
    },
    
    # =========================================================================
    # ORANGE COUNTY
    # =========================================================================
    "ORLANDO": {
        "search_url": "https://permits.orlando.gov/",
        "base_url": "https://www.orlando.gov/Building-Development",
        "deep_link": False,
    },
    "WINTER GARDEN": {
        "search_url": "https://www.cwgdn.com/197/Building-Division",
        "base_url": "https://www.cwgdn.com/197/Building-Division",
        "deep_link": False,
    },
    
    # =========================================================================
    # HILLSBOROUGH COUNTY
    # =========================================================================
    "TAMPA": {
        "search_url": "https://aca-prod.accela.com/TAMPA/Default.aspx",
        "base_url": "https://www.tampa.gov/construction-services",
        "deep_link": False,  # Accela system
    },
    
    # =========================================================================
    # DUVAL COUNTY
    # =========================================================================
    "JACKSONVILLE": {
        "search_url": "https://buildinginspections.coj.net/",
        "base_url": "https://www.coj.net/departments/buildings",
        "deep_link": False,
    },
    
    # =========================================================================
    # PALM BEACH COUNTY
    # =========================================================================
    "WEST PALM BEACH": {
        "search_url": "https://www.wpb.org/government/development-services/building-division",
        "base_url": "https://www.wpb.org/government/development-services/building-division",
        "deep_link": False,
    },
    
    # =========================================================================
    # OSCEOLA COUNTY
    # =========================================================================
    "KISSIMMEE": {
        "search_url": "https://www.kissimmee.gov/departments-services/development-services/building-division",
        "base_url": "https://www.kissimmee.gov/departments-services/development-services/building-division",
        "deep_link": False,
    },
    
    # =========================================================================
    # MARTIN COUNTY
    # =========================================================================
    "STUART": {
        "search_url": "https://www.cityofstuart.us/198/Building",
        "base_url": "https://www.cityofstuart.us/198/Building",
        "deep_link": False,
    },
    
    # =========================================================================
    # VOLUSIA COUNTY
    # =========================================================================
    "DAYTONA BEACH": {
        "search_url": "https://www.codb.us/149/Building-Inspections",
        "base_url": "https://www.codb.us/149/Building-Inspections",
        "deep_link": False,
    },
    
    # =========================================================================
    # ESCAMBIA COUNTY
    # =========================================================================
    "PENSACOLA": {
        "search_url": "https://www.cityofpensacola.com/234/Building-Inspection-Services",
        "base_url": "https://www.cityofpensacola.com/234/Building-Inspection-Services",
        "deep_link": False,
    },
    
    # =========================================================================
    # COUNTY-LEVEL (for unincorporated areas)
    # =========================================================================
    "BROWARD": {
        "search_url": "https://www.broward.org/Building/Pages/default.aspx",
        "base_url": "https://www.broward.org/Building/Pages/default.aspx",
        "deep_link": False,
    },
    "MIAMI-DADE": {
        "search_url": "https://www.miamidade.gov/permits/",
        "base_url": "https://www.miamidade.gov/permits/",
        "deep_link": False,
    },
    "ORANGE": {
        "search_url": "https://www.orangecountyfl.net/BuildingPermitting.aspx",
        "base_url": "https://www.orangecountyfl.net/BuildingPermitting.aspx",
        "deep_link": False,
    },
    "HILLSBOROUGH": {
        "search_url": "https://www.hillsboroughcounty.org/residents/property-owners-and-renters/building-and-renovations",
        "base_url": "https://www.hillsboroughcounty.org/residents/property-owners-and-renters/building-and-renovations",
        "deep_link": False,
    },
    "DUVAL": {
        "search_url": "https://buildinginspections.coj.net/",
        "base_url": "https://www.coj.net/departments/buildings",
        "deep_link": False,
    },
    "PALM BEACH": {
        "search_url": "https://discover.pbcgov.org/pzb/building/Pages/default.aspx",
        "base_url": "https://discover.pbcgov.org/pzb/building/Pages/default.aspx",
        "deep_link": False,
    },
}


def get_permit_portal_url(jurisdiction: str, permit_number: str = None) -> str | None:
    """
    Get the permit portal URL for a given jurisdiction.
    
    Args:
        jurisdiction: The jurisdiction name (e.g., "MIAMI BEACH", "FORT LAUDERDALE")
        permit_number: Optional permit number for deep linking (if supported)
    
    Returns:
        URL string or None if jurisdiction not found
    """
    jurisdiction_upper = jurisdiction.upper().strip() if jurisdiction else None
    
    if not jurisdiction_upper:
        return None
    
    portal = PERMIT_PORTALS.get(jurisdiction_upper)
    
    if not portal:
        # Try common variations
        variations = [
            jurisdiction_upper.replace(" COUNTY", ""),
            jurisdiction_upper.replace("CITY OF ", ""),
            jurisdiction_upper.replace("TOWN OF ", ""),
        ]
        for var in variations:
            portal = PERMIT_PORTALS.get(var)
            if portal:
                break
    
    if not portal:
        return None
    
    # If deep linking is supported and we have a permit number, try to build deep link
    if permit_number and portal.get("deep_link"):
        # Add deep link logic here for specific systems (Accela, etc.)
        pass
    
    # Return the search URL (preferred) or base URL
    return portal.get("search_url") or portal.get("base_url")


def get_all_jurisdictions() -> list[str]:
    """Return a list of all supported jurisdiction names."""
    return sorted(PERMIT_PORTALS.keys())


# Quick test
if __name__ == "__main__":
    print("Supported Florida Jurisdictions:")
    print("-" * 50)
    for j in get_all_jurisdictions():
        url = get_permit_portal_url(j)
        print(f"  {j:25} → {url[:50]}...")
    
    print("\n" + "=" * 50)
    print("Test lookups:")
    print("=" * 50)
    
    test_cases = [
        ("MIAMI BEACH", "BCR2504620"),
        ("FORT LAUDERDALE", "BLD-CERT-22080023"),
        ("ORLANDO", "BLD2025-12625"),
        ("Unknown City", "ABC123"),
    ]
    
    for jurisdiction, permit in test_cases:
        url = get_permit_portal_url(jurisdiction, permit)
        print(f"  {jurisdiction:20} → {url or 'NOT FOUND'}")
