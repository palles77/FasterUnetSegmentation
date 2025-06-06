# CopyAndList.ps1
# ----------------------------------------------------------------------------
# Copies files from "parsed_dataset" subdirectories to "parsed_dataset_test"
# without using xcopy/robocopy, then produces a list of .png files.
# ----------------------------------------------------------------------------

# --- 1) Ensure the destination subfolders exist, then Copy from drhagis ---
New-Item -ItemType Directory -Path ".\parsed_dataset_test\ground"   -Force | Out-Null
New-Item -ItemType Directory -Path ".\parsed_dataset_test\original" -Force | Out-Null

Copy-Item ".\parsed_dataset\drhagis\ground\*"   ".\parsed_dataset_test\ground"   -Recurse -Force
Copy-Item ".\parsed_dataset\drhagis\original\*" ".\parsed_dataset_test\original" -Recurse -Force

# --- 2) Copy from hrf ---
Copy-Item ".\parsed_dataset\hrf\ground\*"   ".\parsed_dataset_test\ground"   -Recurse -Force
Copy-Item ".\parsed_dataset\hrf\original\*" ".\parsed_dataset_test\original" -Recurse -Force

# --- 3) Copy from stare ---
Copy-Item ".\parsed_dataset\stare\ground\*"   ".\parsed_dataset_test\ground"   -Recurse -Force
Copy-Item ".\parsed_dataset\stare\original\*" ".\parsed_dataset_test\original" -Recurse -Force

# --- 4) Change directory to ground, list all .png, and save to list.dbl ---
Set-Location ".\parsed_dataset_test\ground"
Get-ChildItem -Name *.png | Out-File -FilePath "..\list.dbl" -Encoding ascii

# --- 5) Go back to parent directory ---
Set-Location "..\.."

Write-Host "All copying done. The file list.dbl has been created in parsed_dataset_test."
