# Define source directories
$trainOriginalDir = "extracted_files\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\Original"
$trainGroundTruthDir = "extracted_files\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\Ground truth"
$testOriginalDir = "extracted_files\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\Original"
$testGroundTruthDir = "extracted_files\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\train\Ground truth"

# Define target directory
$targetDir = "parsed_dataset"
$counttest = 0

Function Convert-DownloadedFiles {
    param (
        [string]$originalDir,
        [string]$groundTruthDir,
        [string]$targetDir,
        [string]$prefix
    )

    # Create the target directory if it doesn't exist
    if (!(Test-Path -Path $targetDir/fives/original)) {
        New-Item -ItemType Directory -Path $targetDir/fives/original | Out-Null
    }
    if (!(Test-Path -Path $targetDir/fives/ground)) {
        New-Item -ItemType Directory -Path $targetDir/fives/ground | Out-Null
    }
    if (!(Test-Path -Path $targetDir/fives/cnn_train/original)) {
        New-Item -ItemType Directory -Path $targetDir/fives/cnn_train/original | Out-Null
    }
    if (!(Test-Path -Path $targetDir/fives/cnn_train/ground)) {
        New-Item -ItemType Directory -Path $targetDir/fives/cnn_train/ground | Out-Null
    }

    # Get files from the Original directory
    $originalFiles = Get-ChildItem -Path $originalDir -File

    foreach ($file in $originalFiles) {
        # Get the base file name without extension
        $fileName = $file.BaseName
        $fileExtension = $file.Extension

        # Define source file paths
        $originalFilePath = Join-Path $originalDir $file.Name
        $groundTruthFilePath = Join-Path $groundTruthDir $fileName$fileExtension

        # Check if the corresponding file exists in Ground truth
        if (Test-Path -Path $groundTruthFilePath) {
            # Define target file paths
            $originalTargetPath = Join-Path $targetDir ("fives/original/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)
            $groundTruthTargetPath = Join-Path $targetDir ("fives/ground/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)

            # Copy the files to the target directory with new names
            if ($prefix -eq "test" -and $counttest -lt 300) {
                $counttest++
                $originalTargetPath = Join-Path $targetDir ("fives/cnn_train/original/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)
                $groundTruthTargetPath = Join-Path $targetDir ("fives/cnn_train/ground/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)
            }
            
            Copy-Item -Path $originalFilePath -Destination $originalTargetPath
            Copy-Item -Path $groundTruthFilePath -Destination $groundTruthTargetPath
        } else {
            Write-Host "Warning: Corresponding file for '$fileName' not found in Ground truth directory."
        }
    }
}

# Call the function with the required parameters
Convert-DownloadedFiles -originalDir $trainOriginalDir -groundTruthDir $trainGroundTruthDir -targetDir $targetDir -prefix train
Convert-DownloadedFiles -originalDir $testOriginalDir -groundTruthDir $testGroundTruthDir -targetDir $targetDir -prefix test

Set-Location "parsed_dataset\fives\cnn_train\ground"
Get-ChildItem -Name *.png | Out-File -FilePath "..\list_all.dbl" -Encoding ascii
Get-Content "..\list_all.dbl" | Select-Object -First 25 | Set-Content "..\list_short.dbl"
Set-Location "..\..\..\.."

Write-Host "File copying and renaming completed!"
