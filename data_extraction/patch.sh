#!/bin/bash

# usage: ./patch.sh /root/coq_train/ /root/coq_train_patch/

SRC_DIR="$1"
PATCH_DIR="$2"

if [ -z "$SRC_DIR" ] || [ -z "$PATCH_DIR" ]; then
    echo "Usage: $0 <src_dir> <patch_dir>"
    exit 1
fi

SRC_DIR="${SRC_DIR%/}"
PATCH_DIR="${PATCH_DIR%/}"

find "$PATCH_DIR" -type f | while read patch_file; do
    src_file="${patch_file/$PATCH_DIR/$SRC_DIR}"
    if [ -f "$src_file" ]; then
        cp "$patch_file" "$src_file"
        echo "Patched $src_file"
    else
        echo "Skip $src_file (not exist)"
    fi
done