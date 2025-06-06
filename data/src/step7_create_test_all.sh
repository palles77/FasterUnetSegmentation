#!/usr/bin/env bash
# copy_and_list.sh
# -------------------------------------------------------
# Copies files from "parsed_dataset" into "parsed_dataset_test"
# subdirectories, then creates a list of .png files in list.dbl.
# -------------------------------------------------------

# Exit on first error:
set -e

# 1) Ensure the destination folders exist
mkdir -p "parsed_dataset_test/ground"
mkdir -p "parsed_dataset_test/original"

# 2) Copy from drhagis -> parsed_dataset_test
cp -r "parsed_dataset/drhagis/ground/"*   "parsed_dataset_test/ground"
cp -r "parsed_dataset/drhagis/original/"* "parsed_dataset_test/original"

# 3) Copy from hrf -> parsed_dataset_test
cp -r "parsed_dataset/hrf/ground/"*   "parsed_dataset_test/ground"
cp -r "parsed_dataset/hrf/original/"* "parsed_dataset_test/original"

# 4) Copy from stare -> parsed_dataset_test
cp -r "parsed_dataset/stare/ground/"*   "parsed_dataset_test/ground"
cp -r "parsed_dataset/stare/original/"* "parsed_dataset_test/original"

# 5) Change directory to ground
cd "parsed_dataset_test/ground"

# 6) List all .png files in bare format and write them to ../list.dbl
ls -1 *.png > ../list.dbl

# 7) Return to parent directory
cd ../..

echo "All copying done. The file list.dbl has been created in parsed_dataset_test."
