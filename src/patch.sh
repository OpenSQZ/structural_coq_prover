#!/bin/bash

# patch.sh - Apply structural Coq prover modifications to Coq 8.11 source
# 
# Usage: ./patch.sh /path/to/coq-8.11-source /path/to/install-prefix
#
# This script patches a clean Coq 8.11 source tree with our modifications
# that enable structured information output.
# You need to manually build and install afterwards.

set -e  # Exit on any error

if [ $# -ne 2 ]; then
    echo "Usage: $0 <coq-source-dir> <install-prefix>"
    echo ""
    echo "Example:"
    echo "  git clone --branch V8.11.2 https://github.com/coq/coq.git coq-8.11"
    echo "  ./patch.sh coq-8.11 /usr/local/coq-structured"
    echo ""
    echo "This will:"
    echo "  1. Copy our modified source files to the Coq source tree"
    echo "  2. Configure Coq with the specified install prefix"
    echo "  3. Build and install the modified Coq"
    exit 1
fi

COQ_SOURCE_DIR="$1"
INSTALL_PREFIX="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR"

# Validate inputs
if [ ! -d "$COQ_SOURCE_DIR" ]; then
    echo "Error: Coq source directory '$COQ_SOURCE_DIR' does not exist"
    exit 1
fi

if [ ! -f "$COQ_SOURCE_DIR/configure" ]; then
    echo "Error: '$COQ_SOURCE_DIR' does not appear to be a Coq source directory"
    echo "Please clone Coq 8.11 first:"
    echo "  git clone --branch V8.11.2 https://github.com/coq/coq.git"
    exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Modified source directory '$SRC_DIR' does not exist"
    exit 1
fi

echo "=== Structural Coq Prover Patcher ==="
echo "Coq source: $COQ_SOURCE_DIR"
echo "Install to: $INSTALL_PREFIX"
echo "Patch from: $SRC_DIR"
echo ""

# Define file mappings (our_file -> coq_relative_path)
declare -A FILE_MAPPINGS=(
    ["cLexer.ml"]="parsing/cLexer.ml"
    ["classes.ml"]="vernac/classes.ml"
    ["comArguments.ml"]="vernac/comArguments.ml"
    ["comAssumption.ml"]="vernac/comAssumption.ml"
    ["comDefinition.ml"]="vernac/comDefinition.ml"
    ["comFixpoint.ml"]="vernac/comFixpoint.ml"
    ["comInductive.ml"]="vernac/comInductive.ml"
    ["comPrimitive.ml"]="vernac/comPrimitive.ml"
    ["comProgramFixpoint.ml"]="vernac/comProgramFixpoint.ml"
    ["constrintern.ml"]="interp/constrintern.ml"
    ["detyping.ml"]="pretyping/detyping.ml"
    ["globEnv.ml"]="pretyping/globEnv.ml"
    ["names.ml"]="kernel/names.ml"
    ["names.mli"]="kernel/names.mli"
    ["nametab.ml"]="library/nametab.ml"
    ["pp.ml"]="lib/pp.ml"
    ["ppconstr.ml"]="printing/ppconstr.ml"
    ["ppconstr.mli"]="printing/ppconstr.mli"
    ["pptactic.ml"]="plugins/ltac/pptactic.ml"
    ["ppvernac.ml"]="vernac/ppvernac.ml"
    ["ppvernac.mli"]="vernac/ppvernac.mli"
    ["pretyping.ml"]="pretyping/pretyping.ml"
    ["pretyping.mli"]="pretyping/pretyping.mli"
    ["proof_global.ml"]="tactics/proof_global.ml"
    ["tacentries.ml"]="plugins/ltac/tacentries.ml"
    ["tacinterp.ml"]="plugins/ltac/tacinterp.ml"
    ["tacinterp.mli"]="plugins/ltac/tacinterp.mli"
    ["typeops.ml"]="kernel/typeops.ml"
    ["vernacentries.ml"]="vernac/vernacentries.ml"
    ["vernacinterp.ml"]="vernac/vernacinterp.ml"
)

# Apply patches
echo "Applying patches..."
PATCHED_COUNT=0

# Get all keys first to avoid iteration issues
mapfile -t files < <(printf '%s\n' "${!FILE_MAPPINGS[@]}")

for our_file in "${files[@]}"; do
    coq_file="${FILE_MAPPINGS[$our_file]}"
    our_path="$SRC_DIR/$our_file"
    coq_path="$COQ_SOURCE_DIR/$coq_file"
    
    if [ ! -f "$our_path" ]; then
        echo "Warning: Our file '$our_path' not found, skipping"
        continue
    fi
    
    if [ ! -f "$coq_path" ]; then
        echo "Warning: Target file '$coq_path' not found in Coq source, skipping"
        continue
    fi
    
    # Apply patch
    echo "  $our_file -> $coq_file"
    cp "$our_path" "$coq_path"
    PATCHED_COUNT=$((PATCHED_COUNT + 1))
done

echo "Applied $PATCHED_COUNT patches"
echo ""

echo ""
echo "=== Patching Complete ==="
echo "Applied $PATCHED_COUNT patches to Coq source"
echo ""
echo "Next steps:"
echo "  cd $COQ_SOURCE_DIR"
echo "  ./configure -prefix $(realpath -m "$INSTALL_PREFIX")"
echo "  make -j \$(nproc)"
echo "  make install"
echo ""
echo "After installation, you can find the modified Coq binaries at:"
echo "  - coqc: $(realpath -m "$INSTALL_PREFIX")/bin/coqc"
echo "  - coqtop: $(realpath -m "$INSTALL_PREFIX")/bin/coqtop"
echo ""
echo "The modified Coq will output structured information including:"
echo "  - All declarations defined in Coq"
echo "  - Type information for definitions"
echo "  - Proof state transitions during verification"