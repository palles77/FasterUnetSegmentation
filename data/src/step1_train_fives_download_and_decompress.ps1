# Define the URL, output path, and extraction path
$url = "https://figshare.com/ndownloader/files/34969398"
$outputPath = "downloaded_fives.rar"
$extractPath = "extracted_files"
$expectedSize = 1764974308

# Function to download the file with progress reporting
Function Download-File {
    param (
        [string]$Url,
        [string]$OutputPath,
        [int]$ExpectedSize
    )

    Write-Host "Downloading file from $Url..."
    $webClient = New-Object System.Net.WebClient

    $downloadCompleted = $false
    $webClient.DownloadFileAsync([Uri]$Url, $OutputPath)

    while (-not $downloadCompleted) {
        Start-Sleep -Milliseconds 500
        if (Test-Path $OutputPath) {
            $fileSize = (Get-Item $OutputPath).Length
            $percentComplete = [math]::Round(($fileSize / $ExpectedSize) * 100, 2)
            Write-Progress -Activity "Downloading file" -Status "$percentComplete% Complete" -PercentComplete $percentComplete
            if ($fileSize -ge $ExpectedSize) {
                $downloadCompleted = $true
            }
        }
    }

    Write-Host "File downloaded successfully to $OutputPath"

    # Validate file size
    $fileSize = (Get-Item $OutputPath).Length
    if ($fileSize -ne $ExpectedSize) {
        Write-Host "Error: Downloaded file size ($fileSize) does not match expected size ($ExpectedSize). Please try downloading again."
        exit 1
    } else {
        Write-Host "File size validation successful."
    }
}

# Function to install Unrar if not already installed
Function Install-Unrar {
    if (-not (Get-Command "unrar" -ErrorAction SilentlyContinue)) {
        Write-Host "Unrar not found. Installing Unrar using winget..."
        winget install --id RARLab.WinRAR -e --source winget --accept-package-agreements --accept-source-agreements
        Write-Host "Unrar installed successfully."
    } else {
        Write-Host "Unrar is already installed."
    }
}

# Function to decompress the file
Function Decompress-File {
    param (
        [string]$FilePath,
        [string]$ExtractPath
    )

    Write-Host "Decompressing file $FilePath..."
    if ($FilePath -like "*.rar") {
        # Decompress .rar files
        Install-Unrar
        & "c:\Program Files\WinRAR\unrar.exe" x -o+ $FilePath $ExtractPath
        Write-Host "File decompressed successfully to $ExtractPath"
    } else {
        Write-Host "Unsupported file format: $FilePath"
    }
}

# Main script execution

# Check if the file already exists and validate its size
if (Test-Path $outputPath) {
    $fileSize = (Get-Item $outputPath).Length
    if ($fileSize -ne $expectedSize) {
        Write-Host "Existing file size ($fileSize) does not match expected size ($expectedSize). Downloading again..."
        Remove-Item $outputPath
        Download-File -Url $url -OutputPath $outputPath -ExpectedSize $expectedSize
    } else {
        Write-Host "Existing file size matches expected size. Skipping download."
    }
} else {
    Download-File -Url $url -OutputPath $outputPath -ExpectedSize $expectedSize
}

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath | Out-Null
}

Decompress-File -FilePath $outputPath -ExtractPath $extractPath