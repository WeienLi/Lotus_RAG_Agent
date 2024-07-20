#!/bin/bash

input_file="../Data/split_car_stats.json"
output_prefix="../Data/split_car_stats_refine"
chunk_size=300

# Array of ranges to process
ranges=(
    "0 200"
    "200 400"
    "400 600"
    "600 800"
    "800 1000"
    "1000 1200"
    "1200 1355"
)

for i in "${!ranges[@]}"; do
    range=(${ranges[$i]})
    start=${range[0]}
    end=${range[1]}
    output_file="${output_prefix}_$((i + 1)).json"

    echo "Processing chunks ${start} to ${end}..."
    python3 refine.py $input_file $output_file $start $end

    if [ $i -lt $((${#ranges[@]} - 1)) ]; then
        echo "Sleeping for 1 minute before processing the next batch..."
        sleep 60
    fi
done
