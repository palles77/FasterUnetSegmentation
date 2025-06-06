#!/usr/bin/env bash

# Define source directories (make sure to quote paths if they contain spaces)
trainOriginalDir="extracted_files/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/train/Original"
trainGroundTruthDir="extracted_files/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/train/Ground truth"

testOriginalDir="extracted_files/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/train/Original"
testGroundTruthDir="extracted_files/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/train/Ground truth"

# Define target directory
targetDir="parsed_dataset"
globaltestcount=0

# ---------------------------------------
# Function: convert_downloaded_files
# ---------------------------------------
# Parameters:
#   1) originalDir
#   2) groundTruthDir
#   3) targetDir
#   4) prefix
#
# This function copies images from the original directory and
# their corresponding ground-truth files from the groundTruthDir
# into a new directory structure under targetDir/fives/{original,ground},
# renaming the files by prefixing them with <prefix>_ (in lowercase).
# ---------------------------------------
convert_downloaded_files() {
  local originalDir="$1"
  local groundTruthDir="$2"
  local targetDir="$3"
  local prefix="$4"

  # Create target subdirectories if they don't exist
  mkdir -p "${targetDir}/fives/original"
  mkdir -p "${targetDir}/fives/ground"
  mkdir -p "${targetDir}/fives/cnn_train/original"
  mkdir -p "${targetDir}/fives/cnn_train/ground"

  # Loop through each file in the original directory
  for file in "${originalDir}"/*; do
    # Skip if it's not a regular file
    [[ -f "$file" ]] || continue

    # Extract filename and extension
    filename="$(basename "$file")"
    basefile="${filename%.*}"       # filename without extension
    extension="${filename##*.}"     # extension without dot
    echo "Processing: $basefile"

    # Construct corresponding ground-truth filepath
    groundTruthFile="${groundTruthDir}/${filename}"

    # Check if ground-truth file exists
    if [[ -f "$groundTruthFile" ]]; then
      # Define target file paths (make basefile lowercase)
      # Note: ${basefile,,} converts to lowercase in Bash 4+

      if [[ $prefix == "test" && $globaltestcount -lt 300 ]]; then
        globaltestcount=$((globaltestcount+1))
        originalTargetPath="${targetDir}/fives/cnn_train/original/${prefix}_${basefile,,}.${extension}"
        groundTruthTargetPath="${targetDir}/fives/cnn_train/ground/${prefix}_${basefile,,}.${extension}"
      else
        originalTargetPath="${targetDir}/fives/original/${prefix}_${basefile,,}.${extension}"
        groundTruthTargetPath="${targetDir}/fives/ground/${prefix}_${basefile,,}.${extension}"
      fi

      # Copy the files
      cp "$file" "$originalTargetPath"
      cp "$groundTruthFile" "$groundTruthTargetPath"
    else
      echo "Warning: Corresponding file for '$basefile' not found in Ground truth directory."
    fi
  done
}

# Call the function with the required parameters
convert_downloaded_files "$trainOriginalDir" "$trainGroundTruthDir" "$targetDir" "train"
convert_downloaded_files "$testOriginalDir" "$testGroundTruthDir" "$targetDir" "test"

cd "${targetDir}/fives/cnn_train/ground"
ls -1 *.png > ../list_all.dbl
head -n 25 ../list_all.dbl > ../list_short.dbl
cd ../../../..

echo "File copying and renaming completed!"
