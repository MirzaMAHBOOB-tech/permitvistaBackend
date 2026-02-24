"""
Florida Permit Portal URLs
Maps city names (as returned by Shovels API) to their online permit portals.

Usage:
    from permit_portals import get_permit_portal_url
    
    url = get_permit_portal_url("MIAMI BEACH")
    # Returns: "https://www.miamibeachfl.gov/city-hall/building/online-permitting/"
"""

# Permit portal configurations
# format: "CITY_NAME": {
#     "search_url": URL for searching permits (preferred),
#     "base_url": General permit portal URL (fallback),
# }

PERMIT_PORTALS = {
    # =========================================================================
    # MIAMI-DADE COUNTY
    # =========================================================================
    "MIAMI": {
        "search_url": "https://www.miamigov.com/My-Government/Departments/Building/Permit-Search",
        "base_url": "https://www.miamigov.com/My-Government/Departments/Building",
    },
    "MIAMI BEACH": {
        "search_url": "https://www.miamibeachfl.gov/city-hall/building/online-permitting/",
        "base_url": "https://www.miamibeachfl.gov/city-hall/building/",
    },
    "CORAL GABLES": {
        "search_url": "https://www.coralgables.com/departments/development-services/building-division/permit-search",
        "base_url": "https://www.coralgables.com/departments/development-services/building-division",
    },
    
    # =========================================================================
    # BROWARD COUNTY
    # =========================================================================
    "FORT LAUDERDALE": {
        "search_url": "https://aca-prod.accela.com/FTLAUDERDALE/Cap/CapHome.aspx?module=Building",
        "base_url": "https://www.fortlauderdale.gov/departments/development-services/building-services",
    },
    "POMPANO BEACH": {
        "search_url": "https://epermits.pompanobeachfl.gov/EnerGov_Prod/SelfService",
        "base_url": "https://www.pompanobeachfl.gov/government/departments/development-services/building-inspections-division",
    },
    "HALLANDALE BEACH": {
        "search_url": "https://www.hallandalebeachfl.gov/149/Building-Services",
        "base_url": "https://www.hallandalebeachfl.gov/149/Building-Services",
    },
    "OAKLAND PARK": {
        "search_url": "https://www.oaklandparkfl.gov/198/Building-Division",
        "base_url": "https://www.oaklandparkfl.gov/198/Building-Division",
    },
    "DANIA BEACH": {
        "search_url": "https://www.daniabeachfl.gov/262/Building-Division",
        "base_url": "https://www.daniabeachfl.gov/262/Building-Division",
    },
    "DEERFIELD BEACH": {
        "search_url": "https://www.deerfield-beach.com/506/Building-Division",
        "base_url": "https://www.deerfield-beach.com/506/Building-Division",
    },
    
    # =========================================================================
    # PALM BEACH COUNTY
    # =========================================================================
    "WEST PALM BEACH": {
        "search_url": "https://www.wpb.org/government/development-services/building-division",
        "base_url": "https://www.wpb.org/government/development-services/building-division",
    },
    "BOCA RATON": {
        "search_url": "https://www.myboca.us/270/Building-Division",
        "base_url": "https://www.myboca.us/270/Building-Division",
    },
    "BOYNTON BEACH": {
        "search_url": "https://www.boynton-beach.org/building",
        "base_url": "https://www.boynton-beach.org/building",
    },
    
    # =========================================================================
    # ORANGE COUNTY
    # =========================================================================
    "ORLANDO": {
        "search_url": "https://permits.orlando.gov/",
        "base_url": "https://www.orlando.gov/Building-Development",
    },
    "KISSIMMEE": {
        "search_url": "https://www.kissimmee.gov/departments-services/development-services/building-division",
        "base_url": "https://www.kissimmee.gov/departments-services/development-services/building-division",
    },
    
    # =========================================================================
    # HILLSBOROUGH COUNTY
    # =========================================================================
    "TAMPA": {
        "search_url": "https://aca-prod.accela.com/TAMPA/Default.aspx",
        "base_url": "https://www.tampa.gov/construction-services",
    },
    
    # =========================================================================
    # PINELLAS COUNTY
    # =========================================================================
    "ST. PETERSBURG": {
        "search_url": "https://egov.stpete.org/portal/",
        "base_url": "https://www.stpete.org/building_and_development/",
    },
    
    # =========================================================================
    # DUVAL COUNTY
    # =========================================================================
    "JACKSONVILLE": {
        "search_url": "https://buildinginspections.coj.net/",
        "base_url": "https://www.coj.net/departments/buildings",
    },
    
    # =========================================================================
    # MARTIN COUNTY
    # =========================================================================
    "STUART": {
        "search_url": "https://www.cityofstuart.us/198/Building",
        "base_url": "https://www.cityofstuart.us/198/Building",
    },
    
    # =========================================================================
    # VOLUSIA COUNTY
    # =========================================================================
    "DAYTONA BEACH": {
        "search_url": "https://www.codb.us/149/Building-Inspections",
        "base_url": "https://www.codb.us/149/Building-Inspections",
    },
    "DELAND": {
        "search_url": "https://www.deland.org/207/Building-Division",
        "base_url": "https://www.deland.org/207/Building-Division",
    },
    "ORMOND BEACH": {
        "search_url": "https://www.ormondbeach.org/157/Building-Division",
        "base_url": "https://www.ormondbeach.org/157/Building-Division",
    },
    
    # =========================================================================
    # BREVARD COUNTY
    # =========================================================================
    "COCOA BEACH": {
        "search_url": "https://www.cityofcocoabeach.com/162/Building-Division",
        "base_url": "https://www.cityofcocoabeach.com/162/Building-Division",
    },
    "MERRITT ISLAND": {
        "search_url": "https://www.brevardfl.gov/BuildingPermit",
        "base_url": "https://www.brevardfl.gov/BuildingPermit",
    },
    
    # =========================================================================
    # ESCAMBIA COUNTY
    # =========================================================================
    "PENSACOLA": {
        "search_url": "https://www.cityofpensacola.com/234/Building-Inspection-Services",
        "base_url": "https://www.cityofpensacola.com/234/Building-Inspection-Services",
    },
    
    # =========================================================================
    # COUNTY-LEVEL FALLBACKS (for unincorporated areas)
    # =========================================================================
    "BROWARD COUNTY": {
        "search_url": "https://www.broward.org/Building/Pages/default.aspx",
        "base_url": "https://www.broward.org/Building/Pages/default.aspx",
    },
    "MIAMI-DADE COUNTY": {
        "search_url": "https://www.miamidade.gov/permits/",
        "base_url": "https://www.miamidade.gov/permits/",
    },
    "PALM BEACH COUNTY": {
        "search_url": "https://discover.pbcgov.org/pzb/building/Pages/default.aspx",
        "base_url": "https://discover.pbcgov.org/pzb/building/Pages/default.aspx",
    },
    "ORANGE COUNTY": {
        "search_url": "https://www.orangecountyfl.net/BuildingPermitting.aspx",
        "base_url": "https://www.orangecountyfl.net/BuildingPermitting.aspx",
    },
    "HILLSBOROUGH COUNTY": {
        "search_url": "https://www.hillsboroughcounty.org/residents/property-owners-and-renters/building-and-renovations",
        "base_url": "https://www.hillsboroughcounty.org/residents/property-owners-and-renters/building-and-renovations",
    },
    "DUVAL COUNTY": {
        "search_url": "https://buildinginspections.coj.net/",
        "base_url": "https://www.coj.net/departments/buildings",
    },
    "VOLUSIA COUNTY": {
        "search_url": "https://www.volusia.org/services/growth-and-resource-management/building-and-zoning/",
        "base_url": "https://www.volusia.org/services/growth-and-resource-management/building-and-zoning/",
    },
    "BREVARD COUNTY": {
        "search_url": "https://www.brevardfl.gov/BuildingPermit",
        "base_url": "https://www.brevardfl.gov/BuildingPermit",
    },
}


def get_permit_portal_url(city: str) -> str | None:
    """
    Get the permit portal URL for a given city.
    
    Args:
        city: The city name (e.g., "MIAMI BEACH", "FORT LAUDERDALE")
    
    Returns:
        URL string or None if city not found
    """
    if not city:
        return None
        
    city_upper = city.upper().strip()
    
    portal = PERMIT_PORTALS.get(city_upper)
    
    if not portal:
        # Try common variations
        variations = [
            city_upper.replace(".", ""),           # ST. PETERSBURG -> ST PETERSBURG
            city_upper.replace("ST ", "ST. "),     # ST PETERSBURG -> ST. PETERSBURG
            city_upper.replace(" BEACH", ""),      # MIAMI BEACH -> MIAMI (fallback)
            city_upper.replace(" COUNTY", ""),     # BROWARD COUNTY -> BROWARD (if key has no COUNTY)
            city_upper + " COUNTY",                # BROWARD -> BROWARD COUNTY
        ]
        for var in variations:
            portal = PERMIT_PORTALS.get(var)
            if portal:
                break
    
    if not portal:
        return None
    
    # Return the search URL (preferred) or base URL (fallback)
    return portal.get("search_url") or portal.get("base_url")


def get_all_cities() -> list[str]:
    """Return a list of all supported city names."""
    return sorted(PERMIT_PORTALS.keys())


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("SUPPORTED FLORIDA CITIES & PERMIT PORTALS")
    print("=" * 60)
    
    for city in get_all_cities():
        url = get_permit_portal_url(city)
        print(f"  {city:25} → {url[:50]}...")
    
    print(f"\n✅ Total cities supported: {len(PERMIT_PORTALS)}")
    
    print("\n" + "=" * 60)
    print("TEST LOOKUPS")
    print("=" * 60)
    
    test_cases = [
        "MIAMI BEACH",
        "FORT LAUDERDALE",
        "ORLANDO",
        "ST. PETERSBURG",
        "BOCA RATON",
        "Unknown City",
    ]
    
    for city in test_cases:
        url = get_permit_portal_url(city)
        status = "✅" if url else "❌"
        print(f"  {status} {city:20} → {url or 'NOT FOUND'}")
