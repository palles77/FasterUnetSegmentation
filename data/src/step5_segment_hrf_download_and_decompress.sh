#!/usr/bin/env bash

# Define the URL, output path, and extraction path
url="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
outputPath="downloaded_hrf.zip"
extractPath="extracted_files/hrf"
expectedSize=73317613

# ---------------------------------------
# Function: check_and_install_unzip
# ---------------------------------------
# Checks if 'unzip' is installed; installs it if not found.
# ---------------------------------------
check_and_install_unzip() {
  if ! command -v unzip &> /dev/null; then
    echo "'unzip' not found. Attempting to install..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y unzip
      elif command -v yum &> /dev/null; then
        sudo yum install -y unzip
      else
        echo "No supported package manager found. Please install 'unzip' manually."
        exit 1
      fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
      echo "Please install 'unzip' manually on Windows (e.g., via Chocolatey: 'choco install unzip')."
      exit 1
    else
      echo "Unsupported OS. Please install 'unzip' manually."
      exit 1
    fi
  else
    echo "'unzip' is already installed."
  fi
}

# ---------------------------------------
# Function: download_file
# ---------------------------------------
# Checks if the specified file already exists.
# If not, downloads it from the provided URL with progress reporting.
# ---------------------------------------
download_file() {
  local url="$1"
  local outputPath="$2"

  echo "Checking if file '$outputPath' already exists..."
  if [[ -f "$outputPath" ]]; then
    echo "File '$outputPath' already exists. Skipping download."
  else
    echo "Downloading file from '$url'..."
    # Show progress bar with curl
    curl -L --progress-bar -o "$outputPath" "$url"
    echo "File downloaded successfully to '$outputPath'."
  fi
}

# ---------------------------------------
# Function: decompress_file
# ---------------------------------------
# Decompresses the given .zip file to the specified folder with progress reporting.
# ---------------------------------------
decompress_file() {
  local filePath="$1"
  local extractPath="$2"

  echo "Decompressing file '$filePath'..."
  # Using unzip with verbose output for progress
  unzip -o "$filePath" -d "$extractPath" | while read -r line; do
    echo "  $line"
  done
  echo "File decompressed successfully to '$extractPath'."
}

# ---------------------------------------
# Main Script Execution
# ---------------------------------------

# Check for unzip and install if necessary
check_and_install_unzip

# 1) Download the file if needed
download_file "$url" "$outputPath"

# 2) Create the target directory if it doesn't exist
if [[ ! -d "$extractPath" ]]; then
  mkdir -p "$extractPath"
fi

# 3) Decompress the zip file
decompress_file "$outputPath" "$extractPath"