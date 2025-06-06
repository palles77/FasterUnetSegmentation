# Train dataset
src/step1_train_fives_download_and_decompress.sh
src/step2_train_fives_parse_downloaded.sh
# Segment datasets
src/step3_segment_drhagis_download_and_decompress.sh
src/step4_segment_drhagis_parse_downloaded.sh
src/step5_segment_hrf_download_and_decompress.sh
src/step6_segment_hrf_parse_downloaded.sh
rm -f cookies.txt
# Create test set all
src/step7_create_test_all.sh