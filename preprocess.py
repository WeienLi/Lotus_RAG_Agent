import json
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    processed_data = []
    content_to_remove = "40/50s\n60s\n70s\n80s\n90s\n00s\n10s\n20s"
    removal_tags = ["\nLTM2", "\nSZ22061112"]
    tag_to_remove_even = "\nLOTUS TheMarque Series 2"
    unknown_char = ""

    for entry in data:
        page_number = entry["page_number"]
        content = entry["content"]
        
        # Remove everything beyond the first occurrence of "\nLTM2" or "\nSZ22061112" including the tags
        for tag in removal_tags:
            tag_index = content.find(tag)
            if tag_index != -1:
                content = content[:tag_index]

        # Subtract 1 from the page number
        page_number -= 1

        # Remove entries for pages 3, 6, and pages from 341 onward
        if page_number not in [2, 5] and page_number < 340:
            # For even pages after subtraction, remove "\nLOTUS TheMarque Series 2"
            if page_number % 2 == 0:
                content = content.replace(tag_to_remove_even, "")

            # For odd pages after subtraction, delete the specified content
            if page_number % 2 != 0:
                if "\n" + content_to_remove + "\n" in content:
                    content = content.replace("\n" + content_to_remove + "\n", "\n")
                elif "\n" + content_to_remove in content:
                    content = content.replace("\n" + content_to_remove, "\n")
                else:
                    content = content.replace(content_to_remove, "")

            # Locate and replace the page number format with a newline
            content = re.sub(rf"(\n?{page_number}\n)", "\n", content)
                
            processed_data.append({"page_number": page_number, "content": content})

    return processed_data


def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file_path = "./best.json"
    output_file_path = "./processed_best.json"

    data = load_json(input_file_path)
    print("JSON data loaded.")

    processed_data = preprocess_data(data)
    print("Data preprocessing complete.")

    save_json(processed_data, output_file_path)
    print(f"Processed data has been saved to {output_file_path}")