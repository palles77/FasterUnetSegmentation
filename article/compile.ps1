# Define the path to the LaTeX file
$latexFilePath = "article.latex"

# Define the output directory
$outputDir = "."

# Add MiKTeX bin directory to PATH
$miktexBinPath = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin"
if (-not ($env:PATH -contains $miktexBinPath)) {
    $env:PATH = "$miktexBinPath;$env:PATH"
}

# Function to check if pdflatex is installed
function Test-Pdflatex {
    $pdflatexPath = (Get-Command pdflatex -ErrorAction SilentlyContinue).Path
    return [bool]$pdflatexPath
}

# Function to install MiKTeX using winget
function Install-MiKTeX {
    Write-Output "Installing MiKTeX using winget..."
    Start-Process -FilePath "winget" -ArgumentList "install -e --id MiKTeX.MiKTeX" -Wait
}

# Check if pdflatex is installed, if not, install MiKTeX
if (-not (Test-Pdflatex)) {
    Write-Output "pdflatex not found. Installing MiKTeX..."
    Install-MiKTeX
    if (-not (Test-Pdflatex)) {
        Write-Output "Failed to install MiKTeX. Exiting."
        exit 1
    }
}

# Set TEXINPUTS to include the current directory
$env:TEXINPUTS = ".;$env:TEXINPUTS"

# Change to the output directory
Set-Location -Path $outputDir

# Delete existing output files
$filesToDelete = @("article.out", "article.pdf", "article.log", "article.bbl", "article.blg", "article.aux")
foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Remove-Item $file
    }
}

# Compile the LaTeX file to PDF
pdflatex -interaction=nonstopmode -output-directory figures figures\image1generalapproach.latex
pdflatex -interaction=nonstopmode -output-directory $outputDir $latexFilePath
bibtex (Get-Item -Path $latexFilePath).BaseName
pdflatex -interaction=nonstopmode -output-directory $outputDir $latexFilePath
pdflatex -interaction=nonstopmode -output-directory $outputDir $latexFilePath


# Check if the PDF was created successfully
$pdfFilePath = [System.IO.Path]::ChangeExtension($latexFilePath, ".pdf")
if (Test-Path $pdfFilePath) {
    Write-Output "PDF compiled successfully: $pdfFilePath"
} else {
    Write-Output "Failed to compile PDF."
}