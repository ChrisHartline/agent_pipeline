#!/bin/bash
# Run connection tests for Clara memory system
#
# Usage:
#   1. Fill in lily_memory/.env with your credentials
#   2. Run: ./tests/run_connection_tests.sh
#
# Or pass credentials directly:
#   SUPABASE_KEY="your-key" FALKORDB_HOSTED_KEY="your-key" ./tests/run_connection_tests.sh

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Load .env if it exists
if [ -f lily_memory/.env ]; then
    echo "Loading credentials from lily_memory/.env..."
    set -a
    source lily_memory/.env
    set +a
fi

# Check required dependencies
echo "Checking dependencies..."
python -c "import supabase" 2>/dev/null || { echo "Installing supabase..."; pip install supabase -q; }
python -c "import falkordb" 2>/dev/null || { echo "Installing falkordb..."; pip install falkordb -q; }
python -c "import cryptography" 2>/dev/null || { echo "Installing cryptography..."; pip install cryptography -q; }

echo ""
echo "Running connection tests..."
echo ""

python tests/test_connections.py
