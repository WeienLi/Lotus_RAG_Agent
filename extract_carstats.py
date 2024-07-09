import json
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_car_stats(content):
    car_stats_pattern = re.compile(
        r'(?:\n)?(?:Milestone Car\n)?MODEL\n.*?\nNAME[-/]FORMULA\n.*?\nYEARS OF PRODUCTION\n.*?\nEXAMPLES BUILT\n.*?\nENGINE TYPE(?:[-/]\s?SIZE)?\n.*?\n(?:ENGINE SIZE[-/]\s?POWER\n.*?\n)?LENGTH\s?[-/]\s?WIDTH\s?[-/]\s?HEIGHT.*?\nWHEELBASE\n.*?\nWEIGHT\s?.*?(?=\n|$)',
        re.DOTALL
    )
    car_stats_list = []
    while True:
        match = car_stats_pattern.search(content)
        if not match:
            break
        car_stats = match.group(0).strip()
        car_stats_list.append(car_stats)
        content = content[:match.start()] + content[match.end():]

    return content.strip(), car_stats_list

def preprocess_data(data):
    processed_data = []

    for entry in data:
        page_number = entry["page_number"]
        content = entry["content"]
        
        # Manually handle page 93
        if page_number == 93:
            specific_pattern = re.compile(
                r'\nMODEL\s*Type\s*50\nNAME/FORMULA\s*Lotus\s*\(Elan\)\s*\+2S\nYEARS OF PRODUCTION\s*1969-71\nEXAMPLES BUILT\s*3576\nENGINE TYPE\s*Lotus-Ford Twin-cam\nENGINE SIZE/POWER\s*1588cc/118bhp\nLENGTH/WIDTH/HEIGHT\s*168in/64in/48in\nWHEELBASE\s*96in\nWEIGHT\s*1960-1980lb/884-898kg',
                re.DOTALL
            )
            match = specific_pattern.search(content)
            if match:
                car_stats = match.group(0).strip()
                content = content[:match.start()] + content[match.end():]
                new_entry = {
                    "page_number": page_number,
                    "content": content.strip(),
                    "car_stats": [car_stats]
                }
            else:
                new_entry = {
                    "page_number": page_number,
                    "content": content.strip()
                }
        else:
            # Extract car stats
            content, car_stats_list = extract_car_stats(content)
            
            # Prepare the new entry
            new_entry = {
                "page_number": page_number,
                "content": content
            }
            if car_stats_list:
                new_entry["car_stats"] = car_stats_list
        
        processed_data.append(new_entry)

    return processed_data

def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file_path = "./processed_best.json"
    output_file_path = "./car_stats.json"

    data = load_json(input_file_path)
    print("JSON data loaded.")

    processed_data = preprocess_data(data)
    print("Data preprocessing complete.")

    save_json(processed_data, output_file_path)
    print(f"Processed data with car stats has been saved to {output_file_path}")