#!/usr/bin/env python3
"""
Simple debug script for database search issues
Run with: python simple_debug.py
"""
import os
import pyodbc
import re

# Database configuration (copy from api_server.py)
DB_SERVER = os.getenv("DB_SERVER", "permitvista-db.database.windows.net")
DB_DATABASE = os.getenv("DB_DATABASE", "free-sql-db-7590410")
DB_USER = os.getenv("DB_USER", "permitvistaadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def get_connection_string():
    """Build connection string"""
    return (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

def normalize_text(s: str) -> str:
    """Normalize text for searching"""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def extract_address_components(input_address: str):
    """Extract address components"""
    if not input_address:
        return None, None, None

    s = input_address.strip()
    zip_match = re.search(r"(\d{5})(?:-\d{4})?$", s)
    zip_code = zip_match.group(1) if zip_match else None

    num_match = re.match(r"^\s*(\d+)\b", s)
    street_number = num_match.group(1) if num_match else None

    route = s
    if street_number:
        route = route[len(street_number):].strip()

    if "," in route:
        route = route.split(",")[0].strip()

    route_norm = normalize_text(route)

    return street_number, route_norm if route_norm else None, zip_code

def test_connection():
    """Test database connection"""
    print("Testing database connection...")
    try:
        conn = pyodbc.connect(get_connection_string())
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_table_data():
    """Check what data exists in tables"""
    print("\nChecking table data...")

    try:
        conn = pyodbc.connect(get_connection_string())
        cursor = conn.cursor()

        tables = ["dbo.permits", "dbo.miami_permits", "dbo.orlando_permits"]

        for table in tables:
            try:
                # Count records
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"\n{table}: {count} records")

                # Sample addresses
                cursor.execute(f"""
                    SELECT TOP 3 SearchAddress, OriginalAddress1, AddressDescription
                    FROM {table}
                    WHERE SearchAddress IS NOT NULL AND LEN(SearchAddress) > 0
                """)

                print("Sample addresses:")
                for row in cursor.fetchall():
                    print(f"  '{row[0]}' | '{row[1]}' | '{row[2]}'")

            except Exception as e:
                print(f"Error checking {table}: {e}")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")

def test_search():
    """Test search for the problematic address"""
    test_address = "506 N Lincoln Ave Tampa Fl 33609, USA"
    print(f"\nTesting search for: '{test_address}'")

    # Parse address
    street_num, route_text, zip_code = extract_address_components(test_address)
    print(f"Parsed: street_num='{street_num}', route='{route_text}', zip='{zip_code}'")

    try:
        conn = pyodbc.connect(get_connection_string())
        cursor = conn.cursor()

        table = "dbo.permits"

        # Test different search patterns
        search_patterns = []

        if street_num and route_text:
            base_search = f"{street_num} {route_text}"
            search_patterns.append(f"%{base_search}%")

            # Try with directions
            for direction in ['N', 'S', 'E', 'W']:
                dir_search = f"{street_num} {direction} {route_text}"
                search_patterns.append(f"%{dir_search}%")

        if zip_code:
            search_patterns.append(f"%{zip_code}%")

        print(f"Testing {len(search_patterns)} search patterns...")

        for pattern in search_patterns:
            query = f"""
                SELECT COUNT(*) FROM {table}
                WHERE SearchAddress LIKE ?
                   OR OriginalAddress1 LIKE ?
                   OR AddressDescription LIKE ?
            """

            cursor.execute(query, (pattern, pattern, pattern))
            count = cursor.fetchone()[0]

            if count > 0:
                print(f"✅ Pattern '{pattern}' found {count} matches")

                # Show first match
                cursor.execute(f"""
                    SELECT TOP 1 SearchAddress, OriginalAddress1, AddressDescription
                    FROM {table}
                    WHERE SearchAddress LIKE ?
                       OR OriginalAddress1 LIKE ?
                       OR AddressDescription LIKE ?
                """, (pattern, pattern, pattern))

                row = cursor.fetchone()
                if row:
                    print(f"   Sample: '{row[0]}' | '{row[1]}' | '{row[2]}'")
            else:
                print(f"❌ Pattern '{pattern}' found 0 matches")

        conn.close()

    except Exception as e:
        print(f"Search test error: {e}")

if __name__ == "__main__":
    print("PermitVista Search Debug Tool")
    print("=" * 40)

    if not DB_PASSWORD:
        print("❌ DB_PASSWORD not set in environment")
        exit(1)

    if test_connection():
        check_table_data()
        test_search()

    print("\nDebug complete.")
