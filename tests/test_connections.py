#!/usr/bin/env python3
"""
Test script for Supabase and FalkorDB connections.

Usage:
    python tests/test_connections.py

Environment variables:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_KEY or SUPABASE_SERVICEROLE_KEY: Supabase API key

    FALKORDB_HOST: FalkorDB host
    FALKORDB_PORT: FalkorDB port (default: 6379)
    FALKORDB_USER: FalkorDB username (default: falkordb for cloud)
    FALKORDB_HOSTED_KEY: FalkorDB password
    FALKORDB_SSL: Set to 'true' for cloud (auto-detected)
"""

import os
import sys
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_config(name: str, value: str, is_secret: bool = False):
    """Print a configuration value."""
    if is_secret and value:
        display = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
    else:
        display = value or "(not set)"
    print(f"  {name}: {display}")


def test_supabase():
    """Test Supabase connection."""
    print_header("Testing Supabase Connection")

    # Check environment
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICEROLE_KEY") or os.environ.get("SUPABASE_KEY")

    print("\nConfiguration:")
    print_config("SUPABASE_URL", url)
    print_config("SUPABASE_KEY", key, is_secret=True)

    if not url or not key:
        print("\n[ERROR] Missing required environment variables")
        print("  Set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICEROLE_KEY)")
        return False

    try:
        from supabase import create_client
        print("\n[OK] supabase package installed")
    except ImportError:
        print("\n[ERROR] supabase package not installed")
        print("  Install with: pip install supabase")
        return False

    # Test connection
    print("\nConnecting to Supabase...")
    try:
        client = create_client(url, key)
        print("[OK] Client created")

        # Test table access
        print("\nTesting table access (clara_memories)...")
        result = client.table("clara_memories").select("*").limit(1).execute()

        print(f"[OK] Table accessible")
        print(f"  Found {len(result.data)} existing record(s)")

        # Test insert
        print("\nTesting write access...")
        test_id = f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        test_data = {
            "id": test_id,
            "user_id": "connection-test",
            "content": "Test memory from connection test",
            "tier": "session",
            "importance": 0.5,
            "encrypted": False,
            "metadata": {"test": True}
        }

        insert_result = client.table("clara_memories").insert(test_data).execute()
        print(f"[OK] Insert successful (id: {test_id})")

        # Clean up test record
        delete_result = client.table("clara_memories").delete().eq("id", test_id).execute()
        print(f"[OK] Cleanup successful")

        print("\n" + "-" * 40)
        print("  SUPABASE CONNECTION: SUCCESS")
        print("-" * 40)
        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")

        # Provide helpful hints
        error_str = str(e).lower()
        if "invalid api key" in error_str or "apikey" in error_str:
            print("\n  Hint: Check your API key is correct")
            print("  - Use the 'anon' key for public access or 'service_role' for full access")
        elif "relation" in error_str or "does not exist" in error_str:
            print("\n  Hint: The 'clara_memories' table may not exist")
            print("  Run this SQL in Supabase SQL Editor:")
            print("    CREATE TABLE clara_memories (...)")
        elif "rls" in error_str or "policy" in error_str:
            print("\n  Hint: Row Level Security (RLS) may be blocking access")
            print("  Try using SUPABASE_SERVICEROLE_KEY instead of anon key")

        return False


def test_falkordb():
    """Test FalkorDB connection."""
    print_header("Testing FalkorDB Connection")

    # Check environment
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = os.environ.get("FALKORDB_PORT", "6379")
    username = os.environ.get("FALKORDB_USER", "falkordb")
    password = os.environ.get("FALKORDB_HOSTED_KEY")
    ssl_env = os.environ.get("FALKORDB_SSL", "")

    # Auto-detect SSL
    if ssl_env.lower() in ("true", "1", "yes"):
        use_ssl = True
    elif ssl_env.lower() in ("false", "0", "no"):
        use_ssl = False
    else:
        use_ssl = host != "localhost" and not host.startswith("127.")

    print("\nConfiguration:")
    print_config("FALKORDB_HOST", host)
    print_config("FALKORDB_PORT", port)
    print_config("FALKORDB_USER", username)
    print_config("FALKORDB_HOSTED_KEY", password, is_secret=True)
    print_config("SSL", "enabled" if use_ssl else "disabled")

    if not password and host != "localhost":
        print("\n[WARNING] FALKORDB_HOSTED_KEY not set (required for cloud)")

    try:
        from falkordb import FalkorDB
        print("\n[OK] falkordb package installed")
    except ImportError:
        print("\n[ERROR] falkordb package not installed")
        print("  Install with: pip install falkordb")
        return False

    # Test connection
    print(f"\nConnecting to FalkorDB at {host}:{port}...")
    try:
        db = FalkorDB(
            host=host,
            port=int(port),
            username=username,
            password=password,
            ssl=use_ssl
        )
        print("[OK] Connection established")

        # Test graph operations
        print("\nTesting graph operations...")
        graph = db.select_graph("connection_test")
        print("[OK] Graph selected")

        # Create test node
        print("\nCreating test node...")
        graph.query(
            "CREATE (n:TestNode {name: $name, timestamp: $ts})",
            {"name": "connection_test", "ts": datetime.now().isoformat()}
        )
        print("[OK] Node created")

        # Query test node
        result = graph.query(
            "MATCH (n:TestNode {name: $name}) RETURN n",
            {"name": "connection_test"}
        )
        print(f"[OK] Query successful (found {len(result.result_set)} node(s))")

        # Clean up
        graph.query("MATCH (n:TestNode {name: 'connection_test'}) DELETE n")
        print("[OK] Cleanup successful")

        print("\n" + "-" * 40)
        print("  FALKORDB CONNECTION: SUCCESS")
        print("-" * 40)
        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")

        # Provide helpful hints
        error_str = str(e).lower()
        if "invalid username-password" in error_str or "authentication" in error_str:
            print("\n  Hint: Authentication failed")
            print("  - Verify FALKORDB_HOSTED_KEY is correct")
            print("  - For FalkorDB Cloud, get the password from the dashboard")
            print("  - Username is typically 'falkordb' for cloud instances")
        elif "connection refused" in error_str:
            print("\n  Hint: Cannot reach the server")
            print("  - Check FALKORDB_HOST and FALKORDB_PORT")
            print("  - For cloud, ensure SSL is enabled (FALKORDB_SSL=true)")
        elif "ssl" in error_str or "certificate" in error_str:
            print("\n  Hint: SSL/TLS error")
            print("  - Try setting FALKORDB_SSL=true for cloud connections")

        return False


def test_encryption():
    """Test encryption module."""
    print_header("Testing Encryption Module")

    try:
        from lily_memory.encryption import MemoryEncryption, get_encryption, CRYPTO_AVAILABLE
        print(f"\n[OK] Encryption module imported")
        print(f"  cryptography available: {CRYPTO_AVAILABLE}")
    except ImportError as e:
        print(f"\n[ERROR] Could not import encryption module: {e}")
        return False

    if not CRYPTO_AVAILABLE:
        print("\n[WARNING] cryptography package not installed")
        print("  Encryption will be disabled")
        print("  Install with: pip install cryptography")
        return True  # Not a failure, just a warning

    # Check for encryption key
    key_env = os.environ.get("CLARA_MEMORY_KEY")
    print_config("CLARA_MEMORY_KEY", key_env, is_secret=True)

    # Test key generation
    print("\nTesting key generation...")
    key_bytes, key_b64 = MemoryEncryption.generate_key()
    print(f"[OK] Generated key: {key_b64[:20]}...")

    # Test encryption/decryption
    print("\nTesting encrypt/decrypt cycle...")
    enc = MemoryEncryption(key_bytes)

    plaintext = "This is a test memory about my friend Alice in New York."
    encrypted = enc.encrypt(plaintext)
    decrypted = enc.decrypt(encrypted)

    assert decrypted == plaintext, "Decryption failed!"
    print("[OK] Encryption/decryption works correctly")
    print(f"  Original: {plaintext[:40]}...")
    print(f"  Encrypted: {encrypted[:40]}...")
    print(f"  Decrypted: {decrypted[:40]}...")

    if not key_env:
        print("\n[NOTICE] No CLARA_MEMORY_KEY set")
        print("  Generate one with:")
        print(f'    export CLARA_MEMORY_KEY="{key_b64}"')

    print("\n" + "-" * 40)
    print("  ENCRYPTION: SUCCESS")
    print("-" * 40)
    return True


def main():
    """Run all connection tests."""
    print("\n" + "=" * 60)
    print("  CLARA MEMORY SYSTEM - CONNECTION TESTS")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().isoformat()}")

    results = {}

    # Test encryption first (it's a dependency)
    results["encryption"] = test_encryption()

    # Test Supabase
    results["supabase"] = test_supabase()

    # Test FalkorDB
    results["falkordb"] = test_falkordb()

    # Summary
    print_header("TEST SUMMARY")

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[X]"
        print(f"  {symbol} {name.upper()}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
