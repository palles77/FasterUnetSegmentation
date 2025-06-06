# Datasets

This directory's purpose is to store data in two categories:
1. Train data for UNET training in a single resolution
2. Segmentation data for UNET segmentation algorithm testing in resolutions different than UNET

# Download all
You can download and parse all either on Linux or Windows as shown below:
1. On Windows open command line and run from this directory
```
powershell
.\download_windows.ps1
```
2. On Ubuntu Linux open terminal and run from this directory
```
bash download_linux.sh
```

## Training databases

1. FiveS database for training
* Download manually from https://figshare.com/ndownloader/files/3496939
* Windows download automatically and parse:
```
./src/step1_train_fives_download_and_decompress.ps1
./src/step1_train_fives_parse_downloaded.ps1
```
* Linux download automatically and parse:
```
bash src/step1_train_fives_download_and_decompress.sh
bash src/step1_train_fives_parse_downloaded.sh
```
## Segmentation databases

1. Dr Hagis database for segmentation training:
* Download manually from http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip
* Windows download automatically and parse:
```
./src/step3_segment_drhagis_download_and_decompress.ps1
./src/step4_segment_drhagis_parse_downloaded.ps1
```
* Linux download automatically and parse:
```
bash src/step3_segment_drhagis_download_and_decompress.sh
bash src/step4_segment_drhagis_parse_downloaded.sh
```
2. HRF database for segmentation testing:
* Download manually from https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip
* Windows download automatically and parse:
```
./src/step5_segment_hrf_download_and_decompress.ps1
./src/step6_segment_hrf_parse_downloaded.ps1
```
* Linux download automatically and parse:
```
bash src/step5_segment_hrf_download_and_decompress.sh
bash src/step6_segment_hrf_parse_downloaded.sh1
```
3. STARE database for segmentation testing
* It is already manually downloaded in [parsed_dataset/stare](parsed_dataset/stare) directory.
* More details in [parsed_dataset/stare/README.md](parsed_dataset/stare/README.md)
4. All databases into a single 105 files set. Testing DrHagis, HRF and STARE after a model has been created based on FiveS Datasets
* Windows
```
./src/step7_create_test_all.ps1
```
* Linux
```
bash src/step8_create_test_all.sh
```

# Directories

extracted_files - directory where downloaded datasets will be extracted
models - models for vessels training (results from the article)
parsed_dataset - directory with where downloaded dataset will be parsed into with correct structure
segmented - results of experiment 1 and experiment 2 from article
segmented_simple - results of experiment 3 from article
src - scripts for downloading datasets

# Last Update
2025/05/28