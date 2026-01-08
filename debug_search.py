#!/usr/bin/env python3
"""
Debug script to test database search functionality
Run with: python debug_search.py
"""
import os
import sys
import logging

# Add the current directory to path so we can import api_server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_server import get_db_connection, extract_address_components, normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_address_parsing():
    """Test address parsing functions"""
    test_addresses = [
        "506 N Lincoln Ave Tampa Fl 33609, USA",
        "506 N Lincoln Ave Tampa FL 33609",
        "506 Lincoln Ave Tampa FL",
        "506 N Lincoln Ave"
    ]

    print("=== ADDRESS PARSING TEST ===")
    for addr in test_addresses:
        street_num, route_text, zip_code = extract_address_components(addr)
        normalized = normalize_text(addr)
        print(f"Input: '{addr}'")
        print(f"  Parsed: street_num='{street_num}', route='{route_text}', zip='{zip_code}'")
        print(f"  Normalized: '{normalized}'")
        print()

def test_database_sample():
    """Test database connection and get sample records"""
    print("=== DATABASE SAMPLE TEST ===")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            tables = ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]

            for table in tables:
                try:
                    print(f"\n--- Table: {table} ---")

                    # Count total records
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total_count = cursor.fetchone()[0]
                    print(f"Total records: {total_count}")

                    # Get sample addresses
                    cursor.execute(f"""
                        SELECT TOP 5
                            SearchAddress, OriginalAddress1, AddressDescription
                        FROM {table}
                        WHERE SearchAddress IS NOT NULL AND LEN(SearchAddress) > 0
                    """)

                    print("Sample addresses:")
                    for row in cursor.fetchall():
                        search_addr, orig_addr1, addr_desc = row
                        print(f"  SearchAddress: '{search_addr}'")
                        print(f"  OriginalAddress1: '{orig_addr1}'")
                        print(f"  AddressDescription: '{addr_desc}'")
                        print()

                except Exception as e:
                    print(f"Error querying {table}: {e}")

    except Exception as e:
        print(f"Database connection error: {e}")

def test_search_simulation():
    """Simulate the search logic"""
    print("=== SEARCH SIMULATION TEST ===")

    test_address = "506 N Lincoln Ave Tampa Fl 33609, USA"
    print(f"Testing search for: '{test_address}'")

    # Parse address components
    street_num, route_text, zip_code = extract_address_components(test_address)
    print(f"Parsed components: street_num='{street_num}', route='{route_text}', zip='{zip_code}'")

    # Simulate search patterns
    search_patterns = []
    if street_num and route_text:
        base_pattern = f"{street_num} {route_text}"
        search_patterns.append(f"%{base_pattern}%")

        # Try with directions
        for direction in ['N', 'S', 'E', 'W']:
            dir_pattern = f"{street_num} {direction} {route_text}"
            search_patterns.append(f"%{dir_pattern}%")

    if zip_code:
        search_patterns.append(f"%{zip_code}%")

    print(f"Search patterns: {search_patterns}")

    # Test against database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            table = "dbo.permits"  # Test on permits table

            for pattern in search_patterns[:3]:  # Test first 3 patterns
                print(f"\nTesting pattern: '{pattern}' on {table}")

                query = f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE SearchAddress LIKE ?
                       OR OriginalAddress1 LIKE ?
                       OR AddressDescription LIKE ?
                """

                cursor.execute(query, (pattern, pattern, pattern))
                count = cursor.fetchone()[0]
                print(f"  Matches found: {count}")

                if count > 0 and count <= 5:
                    # Show actual matches
                    cursor.execute(f"""
                        SELECT TOP 3 SearchAddress, OriginalAddress1, AddressDescription
                        FROM {table}
                        WHERE SearchAddress LIKE ?
                           OR OriginalAddress1 LIKE ?
                           OR AddressDescription LIKE ?
                    """, (pattern, pattern, pattern))

                    print("  Sample matches:")
                    for row in cursor.fetchall():
                        search_addr, orig_addr1, addr_desc = row
                        print(f"    SearchAddress: '{search_addr}'")
                        print(f"    OriginalAddress1: '{orig_addr1}'")
                        print(f"    AddressDescription: '{addr_desc}'")

    except Exception as e:
        print(f"Search simulation error: {e}")

if __name__ == "__main__":
    print("PermitVista Database Search Debug Tool")
    print("=" * 50)

    test_address_parsing()
    test_database_sample()
    test_search_simulation()

    print("\nDebug complete. Check the output above for issues.")
