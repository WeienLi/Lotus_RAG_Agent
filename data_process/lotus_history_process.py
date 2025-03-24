import fitz 
import json
import os
import yaml
import re

def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def extract_text_by_page(file_path):
    """Page by page extract content and store the page_number with its text

    Args:
        file_path (str): The PDF file path which we want to extract

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a document from the JSON file.
    """
    document = fitz.open(file_path)
    pages_content = []
    pages_to_omit = {1, 2, 4, 5, 7, 8} #Table of contents and title pages
    for page_num in range(len(document)):
        page_number = page_num + 1 
        if page_number in pages_to_omit:
            continue
        page = document.load_page(page_num)
        text = page.get_text()
        pages_content.append({"page_number": page_number, "content": text})

    return pages_content

def preprocess_data(data):
    """Preprocess the data by removing unnecssary headline tags, page_number and Book ID.

    Args:
        data (list[dict]): A list of dictionaries where each dictionary represents a document from the JSON file.

    Returns:
        list[dict]: A list of preprocessed dictionaries where each dictionary represents a document from the JSON file.
    """
    processed_data = []
    content_to_remove = "40/50s\n60s\n70s\n80s\n90s\n00s\n10s\n20s"
    removal_tags = ["\nLTM2", "\nSZ22061112"]
    tag_to_remove_even = "\nLOTUS TheMarque Series 2"
    unknown_char = ""

    for entry in data:
        page_number = entry["page_number"]
        content = entry["content"]
        
        # Remove everything beyond the first occurrence of Book Identifaction
        for tag in removal_tags:
            tag_index = content.find(tag)
            if tag_index != -1:
                content = content[:tag_index]
                
        page_number -= 1

        # Remove entries for Publisher Info (Page 3 and Page 6) and all the references (Page 341 onwards).
        
        if page_number not in [2, 5] and page_number < 340:
            # For even pages, remove the even page footer
            if page_number % 2 == 0:
                content = content.replace(tag_to_remove_even, "")

            # For odd pages, remove the magzine header it can come with different format with possible newline in the beginning or in the end
            if page_number % 2 != 0:
                if "\n" + content_to_remove + "\n" in content:
                    content = content.replace("\n" + content_to_remove + "\n", "\n")
                elif "\n" + content_to_remove in content:
                    content = content.replace("\n" + content_to_remove, "\n")
                else:
                    content = content.replace(content_to_remove, "")
            
            #replace page number with new line
            content = re.sub(rf"(\n?{page_number}\n)", "\n", content)
                
            processed_data.append({"page_number": page_number, "content": content})

    return processed_data

def extract_car_stats(content):
    """Extract Car stats from the Content

    Args:
        content (str): Content of a Page

    Returns:
        tuple: 
            str: String type of the content with car_stats commited
            list: A list of car_stats stored in a list (could be multiple car_stats in a page)
    """
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

def preprocess_carstats_data(data):
    """Preporcess the data by extracting and omitting the car_stats from content and add it as a new entry.

    Args:
        data (list[dict]): A list of dictionaries where each dictionary represents a document from the JSON file.

    Returns:
        list[dict]: A list of preprocessed dictionaries where each dictionary represents a document from the JSON file with an extra entry called car_stats.
    """
    processed_data = []

    for entry in data:
        page_number = entry["page_number"]
        content = entry["content"]
        
        # Manually handle page 93 since it is extracted differently
        if page_number == 93:
            specific_pattern = re.compile(
                r'Sensitive INFO',
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
            content, car_stats_list = extract_car_stats(content)
            new_entry = {
                "page_number": page_number,
                "content": content
            }
            if car_stats_list:
                new_entry["car_stats"] = car_stats_list
        
        processed_data.append(new_entry)

    return processed_data

def split_text(text, chunk_size=200, overlap=50):
    """Splits text into chunks of a given size with overlap.

    Args:
        text (str): The input text to be split.
        chunk_size (int, optional): Maximum words per chunk Defaults to 200.
        overlap (int, optional): Number of words overlapping between chunks Defaults to 50.

    Returns:
        list: list of chunks splitted with chunk_size no more than 200 and overlap of 50
    """
    words = re.split(r'(\s+)', text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(''.join(chunk).strip())
        i += (chunk_size - overlap)
    return chunks

def split_all(data, chunk_size=200, overlap=50):
    """Splits the entire dataset with chunk size of 200 and overlap of 50

    Args:
        data (list[dict]): A list of dictionaries where each dictionary represents a document from the JSON file.

    Returns:
        list[dict]: A list of split dictionaries where each dictionary represents a document according to the split
    """
    processed_data = []
    for entry in data:
        content = entry['content']
        if len(re.findall(r'\b\w+\b', content)) > 200: #If content is over 200 require splitting
            chunks = split_text(content, chunk_size, overlap)
            for chunk in chunks:
                #Make sure for all the chunk within the same page have the page_number
                new_entry = {
                    'content': chunk,
                    'page_number': entry['page_number']
                }
                #Make sure for all the chunk within the same page have the car_stats
                if 'car_stats' in entry:
                    new_entry['car_stats'] = entry['car_stats']
                processed_data.append(new_entry)
        else:
            processed_data.append(entry)
    return processed_data

def save_json(data, output_path):
    """Dump text into Json File

    Args:
        data (list[dict]): A list of dictionaries where each dictionary represents a document from the JSON file.
        output_path (str): The path where we want to dump the Jso file to
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def main():
    config_path = "./data_process/config/config.yaml"
    config = load_config(config_path)
    
    output_directory = config["output_dir"]
    os.makedirs(output_directory, exist_ok=True)
    pages_content = extract_text_by_page(config["car_stats_pdf"])
    preprocessed = preprocess_data(pages_content)
    car_stats_extracted = preprocess_carstats_data(preprocessed)
    final = split_all(car_stats_extracted,200,50)

    output_filename = config["output_filename"]
    output_path = os.path.join(output_directory, output_filename)
    save_json(final,output_path)

if __name__ == "__main__":
    main()