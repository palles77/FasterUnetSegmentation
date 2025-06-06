# Define the URL, output path, and extraction path
$url = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
$outputPath = "downloaded_hrf.zip"
$extractPath = "extracted_files/hrf"
$expectedSize = 73317613

# Function to download the file with progress
Function Download-File {
    param (
        [string]$Url,
        [string]$OutputPath
    )

    $fileExists = Test-Path $OutputPath
    Write-Host "Checking if file $OutputPath already exists... and the result is: $fileExists"
    if ($fileExists) {
        Write-Host "File $OutputPath already exists. Skipping download."
    } else {
        Write-Host "Downloading file from $Url..."
        # Use Start-BitsTransfer for progress reporting
        Start-BitsTransfer -Source $Url -Destination $OutputPath
        Write-Host "File downloaded successfully to $OutputPath"
    }
}

# Function to decompress the file with progress
Function Decompress-File {
    param (
        [string]$FilePath,
        [string]$ExtractPath
    )

    Write-Host "Decompressing file $FilePath..."

    # Get the total number of entries in the zip
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($FilePath)
    $total = $zip.Entries.Count
    $count = 0
    foreach ($entry in $zip.Entries) {
        $count++
        Write-Progress -Activity "Extracting files" -Status "$count of $total" -PercentComplete (($count / $total) * 100)
        $targetPath = Join-Path $ExtractPath $entry.FullName
        $parentDir = Split-Path $targetPath -Parent
        if (!(Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir | Out-Null
        }
        if ($entry.Name) {
            $entryStream = $entry.Open()
            $fileStream = [System.IO.File]::Open($targetPath, [System.IO.FileMode]::Create)
            try {
                $entryStream.CopyTo($fileStream)
            } finally {
                $entryStream.Dispose()
                $fileStream.Dispose()
            }
        }
    }
    $zip.Dispose()
    Write-Progress -Activity "Extracting files" -Completed
    Write-Host "File decompressed successfully to $ExtractPath"
}

# Main script execution
Download-File -Url $url -OutputPath $outputPath

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath | Out-Null
}

Decompress-File -FilePath $outputPath -ExtractPath $extractPath