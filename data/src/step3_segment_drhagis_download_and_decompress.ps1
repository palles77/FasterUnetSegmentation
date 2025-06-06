# Define the Google Drive file ID, output path, and extraction path
$googleDriveFileId = "1evUdkjj0tH4Nl2mEvtNz4vtVROfI5CKb"
$outputPath = "downloaded_drhagis.zip"
$extractPath = "extracted_files"
$expectedSize = 10463309

# Function to download a public Google Drive file in 1MB chunks with immediate progress reporting
Function Download-GoogleDriveFile {
    param (
        [string]$FileId,
        [string]$OutputPath,
        [int]$ExpectedSize
    )

    $baseUrl = "https://drive.google.com/uc?export=download&id=$FileId"
    $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession

    $downloadUrl = "https://drive.google.com/uc?export=download&confirm=$confirm&id=$FileId"

    # Now, download the file in a single pass with progress
    $request = [System.Net.HttpWebRequest]::Create($downloadUrl)
    $request.Method = "GET"
    $request.AllowAutoRedirect = $true
    $request.CookieContainer = New-Object System.Net.CookieContainer
    foreach ($cookie in $session.Cookies.GetCookies($downloadUrl)) {
        $request.CookieContainer.Add($cookie)
    }

    $responseStream = $request.GetResponse().GetResponseStream()
    $fileStream = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create)

    $bufferSize = 10KB
    $buffer = New-Object byte[] $bufferSize
    $totalRead = 0

    do {
        $read = $responseStream.Read($buffer, 0, $bufferSize)
        if ($read -gt 0) {
            $fileStream.Write($buffer, 0, $read)
            $totalRead += $read
            if ($ExpectedSize -gt 0) {
                $percent = [math]::Round(($totalRead / $ExpectedSize) * 100, 2)
                Write-Progress -Activity "Downloading file" -Status "$percent% Complete" -PercentComplete $percent
            } else {
                Write-Progress -Activity "Downloading file" -Status "$totalRead bytes downloaded" -PercentComplete 0
            }
        }
    } while ($read -gt 0)

    $fileStream.Close()
    $responseStream.Close()
    Write-Progress -Activity "Downloading file" -Completed

    # Validate file size
    $fileSize = (Get-Item $OutputPath).Length
    if ($ExpectedSize -gt 0 -and $fileSize -ne $ExpectedSize) {
        Write-Host "Error: Downloaded file size ($fileSize) does not match expected size ($ExpectedSize)."
        exit 1
    } else {
        Write-Host "File downloaded successfully to $OutputPath"
    }
}

# Function to decompress the file
Function Decompress-File {
    param (
        [string]$FilePath,
        [string]$ExtractPath
    )

    Write-Host "Decompressing file $FilePath..."
    Expand-Archive -Path $FilePath -DestinationPath $ExtractPath -Force
    Write-Host "File decompressed successfully to $ExtractPath"
}

# Main script execution
$fileExists = Test-Path $outputPath
Write-Host "Checking if file $outputPath already exists... and the result is: $fileExists"
if ($fileExists) {
    $fileSize = (Get-Item $outputPath).Length
    if ($fileSize -ne $expectedSize) {
        Write-Host "Existing file size ($fileSize) does not match expected size ($expectedSize). Downloading again..."
        Remove-Item $outputPath
        Download-GoogleDriveFile -FileId $googleDriveFileId -OutputPath $outputPath -ExpectedSize $expectedSize
    } else {
        Write-Host "File $outputPath already exists and matches expected size. Skipping download."
    }
} else {
    Write-Host "Downloading file from Google Drive..."
    Download-GoogleDriveFile -FileId $googleDriveFileId -OutputPath $outputPath -ExpectedSize $expectedSize
}

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath | Out-Null
}

Decompress-File -FilePath $outputPath -ExtractPath $extractPath