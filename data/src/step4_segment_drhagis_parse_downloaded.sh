#!/usr/bin/env bash

# -----------------------------------------------------------------------
# Define source directories (adjust paths as needed)
segmentOriginalDir="extracted_files/DRHAGIS/Fundus_Images"
segmentGroundTruthDir="extracted_files/DRHAGIS/Manual_Segmentations"

# Define target directory
targetDir="parsed_dataset"

# -----------------------------------------------------------------------
# Function: convert_jpg_to_png
# Uses ImageMagick's `convert` to convert from JPG to PNG
# -----------------------------------------------------------------------
convert_jpg_to_png() {
  local jpgPath="$1"
  local pngPath="$2"

  echo "Converting '$jpgPath' -> '$pngPath'..."
  # Requires ImageMagick to be installed
  convert "$jpgPath" "$pngPath"
}

# -----------------------------------------------------------------------
# Function: convert_downloaded_files
# Mirrors your PowerShell Convert-Downloaded-Files logic
# -----------------------------------------------------------------------
convert_downloaded_files() {
  local originalDir="$1"
  local groundTruthDir="$2"
  local targetDir="$3"
  local prefix="$4"

  # Create target subdirectories if they don't exist
  mkdir -p "${targetDir}/drhagis/original"
  mkdir -p "${targetDir}/drhagis/ground"

  # Loop over all files in the original directory
  for file in "${originalDir}"/*; do
    # Only proceed if $file is a regular file
    [[ -f "$file" ]] || continue

    # basename => e.g., "IMG_123.jpg"
    local filename
    filename="$(basename "$file")"

    # Strip extension => e.g., "IMG_123"
    local fileNameNoExt="${filename%.*}"

    # Always set fileExtension to ".png"
    local fileExtension=".png"

    # This is our original .jpg file
    local originalFilePath="$file"

    # Construct a .png version of that path => e.g., "IMG_123.png"
    local originalFilePngPath="${originalFilePath%.jpg}.png"

    # Convert the .jpg to .png if the .png doesn't already exist
    if [[ ! -f "$originalFilePngPath" ]]; then
      convert_jpg_to_png "$originalFilePath" "$originalFilePngPath"
    fi

    # Construct ground truth filepath => "filename_manual_orig.png"
    local groundTruthFilePath="${groundTruthDir}/${fileNameNoExt}_manual_orig${fileExtension}"

    # Check if ground truth file exists
    if [[ -f "$groundTruthFilePath" ]]; then
      # Define target file paths for final copying
      local originalTargetPath="${targetDir}/drhagis/original/${prefix}_${fileNameNoExt,,}${fileExtension}"
      local groundTruthTargetPath="${targetDir}/drhagis/ground/${prefix}_${fileNameNoExt,,}${fileExtension}"

      # IMPORTANT: This replicates your PowerShell script's behavior
      # -> It copies the .jpg file to a ".png"-named file
      cp "$originalFilePath" "$originalTargetPath"
      cp "$groundTruthFilePath" "$groundTruthTargetPath"
    else
      echo "Warning: Corresponding file for '$fileNameNoExt' not found in Ground truth directory."
    fi
  done
}

# -----------------------------------------------------------------------
# Main script execution
# -----------------------------------------------------------------------

sudo apt-get update
sudo apt-get install imagemagick

convert_downloaded_files \
  "$segmentOriginalDir" \
  "$segmentGroundTruthDir" \
  "$targetDir" \
  "segment"

echo "File copying and renaming completed!"
