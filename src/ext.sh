#!/bin/bash

# Base directories
src_dir=".tmp/outputs/*/0.tar"
dest_dir=".tmp/extracted"

# Find all .tar files and process each one
for src in $src_dir; do
    # Extract the .tar file
    parent=$(dirname "$src")
    parent=$(basename "$parent")
    dest="$dest_dir/$parent"
    mkdir -p "$dest"
    tar -xf "$src" -C "$dest"
done