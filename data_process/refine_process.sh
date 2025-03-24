#!/bin/bash

input_file="./data/car_stats.json"
output_prefix="./data/car_stats_refine"

ranges=(
    "0 1334"
)

api_keys=(
    "Senstive INFO"
)

for i in "${!ranges[@]}"; do
    range=(${ranges[$i]})
    start=${range[0]}
    end=${range[1]}
    api_key=${api_keys[$i]}
    output_file="${output_prefix}_$((i + 2)).json"

    echo "Processing chunks ${start} to ${end} with API key ${api_key}..."
    python3 ./data_process/refine.py $input_file $output_file $start $end $api_key &

done

wait
echo "All processes are complete."
