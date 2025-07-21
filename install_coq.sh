#!/bin/bash
set -e

# Coq Installation Script for Structural Coq Prover
# This script automates the entire Coq installation and configuration process

if [ $# -ne 1 ]; then
    echo "Usage: $0 <install-prefix>"
    echo ""
    echo "Example:"
    echo "  $0 \$HOME/coq-structured"
    echo ""
    echo "This will install the modified Coq to the specified directory"
    echo "and automatically update config.json with the correct paths."
    exit 1
fi

INSTALL_PREFIX="$(realpath -m "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Structural Coq Prover Installation ==="
echo "Install directory: $INSTALL_PREFIX"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Step 1: Clone Coq source
echo "Step 1: Cloning Coq 8.11 source..."
if [ ! -d "coq-8.11" ]; then
    git clone --branch V8.11.2 https://github.com/coq/coq.git coq-8.11
else
    echo "coq-8.11 directory already exists, skipping clone"
fi

# Step 2: Apply patches
echo ""
echo "Step 2: Applying patches..."
./src/patch.sh coq-8.11 "$INSTALL_PREFIX"

# Step 3: Build and install
echo ""
echo "Step 3: Building and installing Coq..."
cd coq-8.11
./configure -prefix "$INSTALL_PREFIX"
make -j $(nproc)
make install
cd ..

# Step 4: Update config.json
echo ""
echo "Step 4: Updating config.json..."
COQC_PATH="$INSTALL_PREFIX/bin/coqc"
COQTOP_PATH="$INSTALL_PREFIX/bin/coqtop"

if [ -f "config.json" ]; then
    # Use Python to update JSON properly
    python -c "
import json
import sys

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    if 'paths' not in config:
        config['paths'] = {}
    
    config['paths']['coqc_path'] = '$COQC_PATH'
    config['paths']['coqtop_path'] = '$COQTOP_PATH'
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('config.json updated successfully')
except Exception as e:
    print(f'Error updating config.json: {e}')
    sys.exit(1)
"
else
    echo "Warning: config.json not found, please create it and add:"
    echo "{"
    echo "  \"paths\": {"
    echo "    \"coqc_path\": \"$COQC_PATH\","
    echo "    \"coqtop_path\": \"$COQTOP_PATH\""
    echo "  }"
    echo "}"
fi

echo ""
echo "=== Installation Complete ==="
echo "Modified Coq installed to: $INSTALL_PREFIX"
echo "Binaries available at:"
echo "  - coqc: $COQC_PATH"
echo "  - coqtop: $COQTOP_PATH"
echo ""
echo "Next steps:"
echo "  1. Set up environment variables (see README)"
echo "  2. Configure data sources in config.json"
echo "  3. Run data extraction: python data_extraction/main.py --mode data_generation"