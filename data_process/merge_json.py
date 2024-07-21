import json
import glob

def merge_and_sort_json(input_pattern, output_file):
    all_data = []

    # Read and merge all JSON files matching the pattern
    for file_name in sorted(glob.glob(input_pattern)):
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)

    # Sort data by 'id'
    sorted_data = sorted(all_data, key=lambda x: int(x['id']))

    # Save the merged and sorted data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
    
    print(f"Merged and sorted data saved to {output_file}")

if __name__ == "__main__":
    input_pattern = "../Data/split_car_stats_refine_*.json"
    output_file = "../Data/merged_sorted_car_stats.json"
    merge_and_sort_json(input_pattern, output_file)
