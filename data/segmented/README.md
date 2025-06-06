# Info
Directory with segmentation results

# Naming

The naming is based on the following understanding:

segmented_900_20_100_20_10_4_64_64_16
900 - number of images were used for U-Net training,
20 - min scale for rough segmenation step 1,
100 - max scale for rough segmentation 1,
20 - step for rough segmentation 1,
10 - range minus plus from optimal scale from rought segmentation 1,
4 - step between ranges in rough segmentation 2,
64 - rough segmentation 1 window sliding step,
64 - rough segmentation 2 window sliding step,
16 - detailed segmentation window sliding step.

# Analysis usage

```
Get-ChildItem -Path . -Directory | ForEach-Object { & "./analyze_segmentation.ps1" -directory $_.FullName }
```

# Best outcomes

1. 
```
Directory Analysis Report
=========================
Directory Name: 100_20_100_20_10_4_128_128_16

Parameters:
-----------
Training Images: 100
Min Rough Segmentation Percentage: 20%
Max Rough Segmentation Percentage: 100%
Rough Segmentation Step: 20%
Optimal Segmentation Range: plus minus 10%
Detailed Segmentation Range: plus minus 4%
Rough Segmentation Window Step: 128
Detailed Segmentation Window Step: 128
Final Optimal Segmentation Window Step: 16

Processing Time Summary:
------------------------
Total Time Elapsed: 3,316.9300 seconds

Segmentation Quality Analysis (Jaccard Distance):
-------------------------------------------------
Total Samples: 150
Average Jaccard Distance: 0.6114
Standard Deviation: 0.0883
Minimum Jaccard Distance: 0.2722
Maximum Jaccard Distance: 0.7705
```
2. 
Directory Analysis Report
```
=========================
Directory Name: 100_20_100_20_10_4_128_64_32

Parameters:
-----------
Training Images: 100
Min Rough Segmentation Percentage: 20%
Max Rough Segmentation Percentage: 100%
Rough Segmentation Step: 20%
Optimal Segmentation Range: plus minus 10%
Detailed Segmentation Range: plus minus 4%
Rough Segmentation Window Step: 128
Detailed Segmentation Window Step: 64
Final Optimal Segmentation Window Step: 32

Processing Time Summary:
------------------------
Total Time Elapsed: 2,704.0500 seconds

Segmentation Quality Analysis (Jaccard Distance):
-------------------------------------------------
Total Samples: 150
Average Jaccard Distance: 0.6125
Standard Deviation: 0.0865
Minimum Jaccard Distance: 0.2714
Maximum Jaccard Distance: 0.7694
```

# Last Update
2024/03/31