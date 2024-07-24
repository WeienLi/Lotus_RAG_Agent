#!/bin/bash

input_file="../Data/split_car_stats.json"
output_prefix="../Data/split_car_stats_refine"

# Array of ranges to process
ranges=(
    # "200 230"
    # "230 260"
    # "260 290"
    #"290 295"
    "295 299"
    # "320 350"
    # "350 380"
    # "380 400"
)

# Array of API keys
api_keys=(
    #"sk-0vPjiNkbFhOBk9yWi7PViLXdlkmgEqbP2CBDTGLh9hF3nWpX"
    #"sk-g69cMGib6zU8GeNUYZP9pXUS4DOXZOZctLHuquRFMauYusDZ"
    #"sk-Cqxo7CoRB8w81E84L3GZMJQBjc0oJWO7XSI3AnncCeChkFaA"
    # "sk-JkVDWMJ8kaE8GAomkoXuD8TONHT5ELOBhikevIf97itdjCof"
    #"sk-UFSPXf2EhdVu9mw6hcj4vQ3rcC1jOdv13IBfNBvy1WETUzmu"
     "sk-vtqu733JfyRnqXj8XGk28I2VPlnGKpEZ9wurIZvOIm1J9jye"
    # "sk-GvqbGYB5moIeOsuo91X2fb1xkGFr8v72pJKsNkV5L7h3T6Nq"
)

for i in "${!ranges[@]}"; do
    range=(${ranges[$i]})
    start=${range[0]}
    end=${range[1]}
    api_key=${api_keys[$i]}
    output_file="${output_prefix}_$((i + 2)).json"

    echo "Processing chunks ${start} to ${end} with API key ${api_key}..."
    python3 refine.py $input_file $output_file $start $end $api_key &

    # if [ $i -lt $((${#ranges[@]} - 1)) ]; then
    #     echo "Sleeping for 1 minute before processing the next batch..."
    #     sleep 60
    # fi
done

# Wait for all background processes to complete
wait
echo "All processes are complete."
