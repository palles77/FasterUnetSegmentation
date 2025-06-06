# Define source directories
$segmentOriginalDir = "extracted_files\DRHAGIS\Fundus_Images"
$segmentGroundTruthDir = "extracted_files\DRHAGIS\Manual_Segmentations"

# Define target directory
$targetDir = "parsed_dataset"

Function Convert-JpgToPng {
    param (
        [string]$jpgPath,
        [string]$pngPath
    )

    Add-Type -AssemblyName System.Drawing
    $image = [System.Drawing.Image]::FromFile($jpgPath)
    $image.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $image.Dispose()
}

Function Convert-Downloaded-Files {
    param (
        [string]$originalDir,
        [string]$groundTruthDir,
        [string]$targetDir,
        [string]$prefix
    )

    # Create the target directory if it doesn't exist
    if (!(Test-Path -Path $targetDir/drhagis/original)) {
        New-Item -ItemType Directory -Path $targetDir/drhagis/original | Out-Null
    }
    if (!(Test-Path -Path $targetDir/drhagis/ground)) {
        New-Item -ItemType Directory -Path $targetDir/drhagis/ground | Out-Null
    }

    # Get files from the Original directory
    $originalFiles = Get-ChildItem -Path $originalDir -File

    foreach ($file in $originalFiles) {
        # Get the base file name without extension
        $fileName = $file.BaseName
        $fileExtension = ".png"

        # Define source file paths
        $originalFilePath = Join-Path $originalDir $file.Name

        # Convert the originalFilePath which is jpg into png
        $originalFilePngPath = $originalFilePath -replace ".jpg", ".png"
        if (!(Test-Path -Path $originalFilePngPath)) {
            Convert-JpgToPng -jpgPath $originalFilePath -pngPath $originalFilePngPath
        }
        # Define the ground truth file path
        $groundTruthFilePath = Join-Path $groundTruthDir ("{0}_manual_orig{1}" -f $fileName, $fileExtension)

        # Check if the corresponding file exists in Ground truth
        if (Test-Path -Path $groundTruthFilePath) {
            # Define target file paths
            $originalTargetPath = Join-Path $targetDir ("drhagis/original/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)
            $groundTruthTargetPath = Join-Path $targetDir ("drhagis/ground/{0}_{1}{2}" -f $prefix, $fileName.ToLower(), $fileExtension)

            # Copy the files to the target directory with new names
            Copy-Item -Path $originalFilePath -Destination $originalTargetPath
            Copy-Item -Path $groundTruthFilePath -Destination $groundTruthTargetPath
        } else {
            Write-Host "Warning: Corresponding file for '$fileName' not found in Ground truth directory."
        }
    }
}

# Call the function with the required parameters
Convert-Downloaded-Files -originalDir $segmentOriginalDir -groundTruthDir $segmentGroundTruthDir -targetDir $targetDir -prefix segment

Write-Host "File copying and renaming completed!"
