#!/bin/bash

DATA_DIR="./data"
REFINED_FILE="$DATA_DIR/car_stats_refine.json"
ORIGINAL_FILE="$DATA_DIR/car_stats.json"
FINAL_DATA_DIR="$DATA_DIR/final_data"
LOTUS_DIR="$FINAL_DATA_DIR/lotus"
LOTUS_CAR_STATS_DIR="$FINAL_DATA_DIR/lotus_car_stats"
LOTUS_BRAND_INFO_DIR="$FINAL_DATA_DIR/lotus_brand_info"

mkdir -p "$LOTUS_DIR" "$LOTUS_CAR_STATS_DIR" "$LOTUS_BRAND_INFO_DIR"

if [[ -f "$REFINED_FILE" ]]; then
    echo "Using refined car stats file."
    cp "$REFINED_FILE" "$LOTUS_DIR/"
    cp "$REFINED_FILE" "$LOTUS_CAR_STATS_DIR/"
else
    echo "Refined file not found. Using original car stats file."
    cp "$ORIGINAL_FILE" "$LOTUS_DIR/"
    cp "$ORIGINAL_FILE" "$LOTUS_CAR_STATS_DIR/"
fi

cp "$DATA_DIR/20-F.json" "$LOTUS_BRAND_INFO_DIR/"
cp "$DATA_DIR/20-F.json" "$LOTUS_DIR/"

cp "$DATA_DIR/LotusTech-F-4A_2023_12_05.json" "$LOTUS_BRAND_INFO_DIR/"
cp "$DATA_DIR/LotusTech-F-4A_2023_12_05.json" "$LOTUS_DIR/"

echo "Data has been organized"

python3 "$DATA_DIR/data_manager.py"

echo "Data has been loaded in Database"