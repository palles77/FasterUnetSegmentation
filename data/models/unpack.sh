#!/usr/bin/env bash

models=(
  "cnn_model_20250214_222252"
  "unet_model_20250212_002522"
  "cnn_model_20250224_222300"
  "unet_model_20250224_183427"
  "cnn_model_20250529_005118"
  "unet_model_20250529_000149"
)

for model in "${models[@]}"; do
  unzip "$model.zip" -d . 

  # Try both possible extracted paths
  if [ -d "data/models/$model" ]; then
    mv "data/models/$model" .
  elif [ -d "$model" ]; then
    echo "$model extracted to ./$model"
  else
    echo "Warning: $model not found after extraction"
  fi
done

rm -rf data