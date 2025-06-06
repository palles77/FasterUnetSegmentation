#!/bin/bash

# Define the URL, output path, and extraction path
url="https://figshare.com/ndownloader/files/34969398"
outputPath="downloaded_fives.rar"
extractPath="extracted_files"
expectedSize=1764974308

# Function to download the file
download_file() {
    local url=$1
    local outputPath=$2
    local expectedSize=$3

    if [ -f "$outputPath" ]; then
        fileSize=$(stat -c%s "$outputPath")
        if [ "$fileSize" -ne "$expectedSize" ]; then
            echo "Existing file size ($fileSize) does not match expected size ($expectedSize). Downloading again..."
            rm "$outputPath"
        else
            echo "File $outputPath already exists and matches expected size. Skipping download."
            return
        fi
    fi

    echo "Downloading file from $url..."
    curl -o "$outputPath" -L -J -O -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -e "https://figshare.com" -b cookies.txt -c cookies.txt "$url" --progress-bar
    if [ $? -eq 0 ]; then
        echo "File downloaded successfully to $outputPath"
    else
        echo "Failed to download file."
        exit 1
    fi

    # Validate file size
    fileSize=$(stat -c%s "$outputPath")
    if [ "$fileSize" -ne "$expectedSize" ]; then
        echo "Error: Downloaded file size ($fileSize) does not match expected size ($expectedSize). Please try downloading again."
        exit 1
    else
        echo "File size validation successful."
    fi
}

# Function to install Unrar if not already installed
install_unrar() {
    if ! command -v unrar &> /dev/null; then
        echo "Unrar not found. Installing Unrar..."
        sudo apt-get update
        sudo apt-get install -y unrar
        if [ $? -eq 0 ]; then
            echo "Unrar installed successfully."
        else
            echo "Failed to install Unrar."
            exit 1
        fi
    else
        echo "Unrar is already installed."
    fi
}

# Function to decompress the file
decompress_file() {
    local filePath=$1
    local extractPath=$2

    echo "Decompressing file $filePath..."
    if [[ "$filePath" == *.rar ]]; then
        install_unrar
        mkdir -p "$extractPath"
        unrar x -o+ "$filePath" "$extractPath"
        if [ $? -eq 0 ]; then
            echo "File decompressed successfully to $extractPath"
        else
            echo "Failed to decompress file."
            exit 1
        fi
    else
        echo "Unsupported file format: $filePath"
    fi
}

# Main script execution
download_file "$url" "$outputPath" "$expectedSize"

# Create the target directory if it doesn't exist
mkdir -p "$extractPath"

decompress_file "$outputPath" "$extractPath"