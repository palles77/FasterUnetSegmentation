param (
    [string]$directory
)

# Ensure the directory exists
if (!(Test-Path $directory -PathType Container)) {
    Write-Host "Directory '$directory' does not exist."
    exit
}

# Parse directory name
$directoryName = Split-Path -Leaf $directory
$parts = $directoryName -split "_"

$parts.Count -ne 10

$reportFile = "./$directoryName`_report.txt"

$reportContent = @"
Directory Analysis Report
=========================
Directory Name: $directoryName

Parameters:
-----------
Training Images: $(if ($parts.Count -ge 10) { $parts[0] } else { "N/A" })
Min Rough Segmentation Percentage (MINRSP): $(if ($parts.Count -ge 10) { "$($parts[1])%" } else { "N/A" })
Max Rough Segmentation Percentage (MAXRSP): $(if ($parts.Count -ge 10) { "$($parts[2])%" } else { "N/A" })
Rough Segmentation Step (RSS): $(if ($parts.Count -ge 10) { "$($parts[3])%" } else { "N/A" })
Detailed Segmentation Range (DSDSR): $(if ($parts.Count -ge 10) { "plus minus $($parts[4])%" } else { "N/A" })
Detailed Segmentation Step (DSS): $(if ($parts.Count -ge 10) { "plus minus $($parts[5])%" } else { "N/A" })
Grow Percentage Overlap For Random Window (GPOFRW): $(if ($parts.Count -ge 10) { "$($parts[6])%" } else { "N/A" })
Percentage Of The Image Covered By Random Windows (PICRW): $(if ($parts.Count -ge 10) { $parts[7] } else { "N/A" })
Multiplier Of PICRW (MPICRW): $(if ($parts.Count -ge 10) { $parts[8] } else { "N/A" })
Jaccard Index Percentage Calculation Rate (JCPCR): $(if ($parts.Count -ge 10) { $parts[9] } else { "N/A" })

"@

# Extract and sum processing time from summary.txt
$summaryFile = "$directory\summary.txt"
$totalTimeElapsed = 0

if (Test-Path $summaryFile) {
    $timeLines = Get-Content $summaryFile | Where-Object { $_ -match "Time elapsed: ([\d\.]+) seconds" }
    if ($timeLines.Count -gt 0) {
        $totalTimeElapsed = ($timeLines | ForEach-Object { [double]($_ -replace "Time elapsed: ([\d\.]+) seconds", '$1') }) -as [double[]] | Measure-Object -Sum | Select-Object -ExpandProperty Sum
    }
} else {
    Write-Host "summary.txt not found in '$directory'."
}

$reportContent += @"

Processing Time Summary:
------------------------
Total Time Elapsed: {0:N4} seconds

"@ -f $totalTimeElapsed

# Extract Jaccard distances from output.txt
$jaccardDistances = @()

if (Test-Path $summaryFile) {
    $jaccardLines = Get-Content $summaryFile | Where-Object { $_ -match "Ground.*Jaccard Distance: ([\d\.]+)" }
    if ($jaccardLines.Count -gt 0) {
        $jaccardDistances = @()
        $jaccardLines | ForEach-Object { 
            if ($_ -match "Jaccard Distance: ([\d\.]+)") {
                $jaccardDistances += [double]$matches[1]
            }
        }
    }
}

if ($jaccardDistances.Count -gt 0) {
    # Compute statistics manually
    $countJaccard = $jaccardDistances.Count
    $sumJaccard = ($jaccardDistances | Measure-Object -Sum).Sum
    $averageJaccard = if ($countJaccard -gt 0) { $sumJaccard / $countJaccard } else { 0 }

    # Calculate Standard Deviation manually
    if ($countJaccard -gt 1) {
        $variance = ($jaccardDistances | ForEach-Object { [math]::Pow($_ - $averageJaccard, 2) } | Measure-Object -Sum).Sum / $countJaccard
        $stdDevJaccard = [math]::Sqrt($variance)
    } else {
        $stdDevJaccard = 0
    }

    $minJaccard = ($jaccardDistances | Measure-Object -Minimum).Minimum
    $maxJaccard = ($jaccardDistances | Measure-Object -Maximum).Maximum

    $reportContent += @"

Segmentation Quality Analysis (Jaccard Distance):
-------------------------------------------------
Total Samples: $countJaccard
Average Jaccard Distance: {0:N4}
Standard Deviation: {1:N4}
Minimum Jaccard Distance: {2:N4}
Maximum Jaccard Distance: {3:N4}

Interpretation:
---------------
- A lower Jaccard Distance means better segmentation accuracy.
- A high standard deviation suggests variability in segmentation performance.
- If the average Jaccard Distance is close to 0, the model performed well.
- If the maximum Jaccard Distance is high, some images were segmented poorly.
"@ -f $averageJaccard, $stdDevJaccard, $minJaccard, $maxJaccard
} else {
    $reportContent += @"

Segmentation Quality Analysis:
------------------------------
No Jaccard Distance data found in output.txt.
"@
}

# Save the report
$reportContent | Out-File -FilePath $reportFile -Encoding utf8

Write-Host "Report generated: $reportFile"
