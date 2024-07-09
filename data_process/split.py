import json
import re

# Load the JSON database
with open('car_stats.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to split text into chunks of 200 words with 50-word overlap
def split_text(text, chunk_size=200, overlap=50):
    words = re.split(r'(\s+)', text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size*2]
        chunks.append(''.join(chunk).strip())
        i += (chunk_size - overlap) * 2
    return chunks

# Process the database
processed_data = []
for entry in data:
    content = entry['content']
    if len(re.findall(r'\b\w+\b', content)) > 200:
        chunks = split_text(content)
        for chunk in chunks:
            new_entry = {
                'content': chunk,
                'page_number': entry['page_number']
            }
            if 'car_stats' in entry:
                new_entry['car_stats'] = entry['car_stats']
            processed_data.append(new_entry)
    else:
        processed_data.append(entry)

# Save the processed database to a new JSON file
output_path = 'split.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print(f"Processed data has been saved to {output_path}")