#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Define source directories
segmentOriginalDir="extracted_files/hrf/images"
segmentGroundTruthDir="extracted_files/hrf/manual1"

# Define target directory
targetDir="parsed_dataset/hrf"

# -----------------------------------------------------------------------------
# Function: convert_jpg_to_png
# Uses ImageMagick's `convert` to convert from JPG to PNG
# -----------------------------------------------------------------------------
convert_jpg_to_png() {
  local jpgPath="$1"
  local pngPath="$2"
  echo "Converting '$jpgPath' to '$pngPath'..."
  convert "$jpgPath" "$pngPath"
}

# -----------------------------------------------------------------------------
# Function: convert_tif_to_png
# Uses ImageMagick's `convert` to convert from TIF to PNG
# -----------------------------------------------------------------------------
convert_tif_to_png() {
  local tifPath="$1"
  local pngPath="$2"
  echo "Converting '$tifPath' to '$pngPath'..."
  convert $tifPath $pngPath
}

# -----------------------------------------------------------------------------
# Function: convert_downloaded_files
# Mirrors your PowerShell "Convert-Downloaded-Files" logic:
#   - Creates output subdirectories if they don't exist
#   - Converts .jpg -> .png and .tif -> .png if needed
#   - Copies them to the target structure with a prefix
# -----------------------------------------------------------------------------
convert_downloaded_files() {
  local originalDir="$1"
  local groundTruthDir="$2"
  local targetDir="$3"

  # Create target subdirectories if they don't exist
  mkdir -p "${targetDir}/original"
  mkdir -p "${targetDir}/ground"

  # Loop through all files in the original directory
  for file in "${originalDir}"/*.{JPG,jpg}; do
    [[ -f "$file" ]] || continue  # Skip if not a regular file
    echo "Processing file: $file"

    # Extract filename (e.g. "ABC123.jpg") and remove extension (e.g. "ABC123")
    local filename
    filename="$(basename "$file")"
    local fileNameNoExt="${filename%.*}"

    local srcOriginalFilePath="${originalDir}/${fileNameNoExt}.jpg"
    local srcGroundFilePath="${groundTruthDir}/${fileNameNoExt}.tif"
    local dstOriginalFilePngPath="${targetDir}/original/${fileNameNoExt}.png"
    local dstGroundTruthFilePath="${targetDir}/ground/${fileNameNoExt}.png"

    if [[ -f "$srcOriginalFilePath" && -f "$srcGroundFilePath" ]]; then
        convert_jpg_to_png "$srcOriginalFilePath" "$dstOriginalFilePngPath"
        convert_tif_to_png "$srcGroundFilePath" "$dstGroundTruthFilePath"
    fi
  done
}

# -----------------------------------------------------------------------------
# Main script execution
# -----------------------------------------------------------------------------
convert_downloaded_files \
  "$segmentOriginalDir" \
  "$segmentGroundTruthDir" \
  "$targetDir"

echo "File copying and renaming completed!"
