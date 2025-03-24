import pdfplumber
import json
import nltk
from nltk.tokenize import sent_tokenize
import yaml
import os

def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def read_specific_pages(pdf_path, start_page, end_page):
    """
    Extracts and cleantext from specific pages of a PDF file

    Args:
        pdf_path (str): Path to the PDF file
        start_page (int): The first page number to extract
        end_page (int): The last page number to extract (inclusive)

    Returns:
        list[tuple[int, str]]: A list of tuples where each tuple contains:
            - page number (int)
            - cleaned extracted text (str) from that page
    """
    with pdfplumber.open(pdf_path) as pdf:
        specific_pages_data = []
        for page_num in range(start_page, end_page + 1):
            page = pdf.pages[page_num - 1]
            text = page.extract_text()
            cleaned_text = clean_text(text)
            specific_pages_data.append((page_num, cleaned_text))
    return specific_pages_data

def clean_text(text):
    """Cleans text by removing header and footer

    Args:
        text (str): The extracted text from a document

    Returns:
        str: The cleaned text with headers and footers removed
    """
    
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not is_header_or_footer(line)]
    cleaned_text = ' '.join(filtered_lines)
    return cleaned_text

def is_header_or_footer(line):
    """Determine if the line contains head or footer

    Args:
        line (str): A line of text to check

    Returns:
        bool: True if it is header or footer by checking if it has header footer phrases else False
    """
    header_footer_phrases = [
        "TABLE OF CONTENTS",
        "Please Consider the Environment Before Printing This Document",
        "Copyright ©"
    ]
    for phrase in header_footer_phrases:
        if phrase in line:
            return True
    return False

def split_text_by_sentences(text, min_word_count=200, max_word_count=250):
    """Splits text into chunks based on sentence boundaries, ensuring each chunk 
    contains between `min_word_count` and `max_word_count` words

    Args:
        text (str): The input text to be split
        min_word_count (int, optional): The minimum number of words required for a chunk. Defaults to 200
        max_word_count (int, optional): The maximum number of words allowed in a chunk. Defaults to 250

    Returns:
        list[str]: A list of text chunks, each containing a valid range of words
    """
    #tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        #if adding sentence exceeds the word_count
        if current_word_count + sentence_word_count > max_word_count:
            # if bigger than min count we add that in
            if current_word_count >= min_word_count:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    #Left with chunks then we add a chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def is_valid_content(content, exclude_phrases):
    """Checks if a given text content is valid

    Args:
        content (str): The text content to check
        exclude_phrases (list[str]): A list of phrases that we want to exclude

    Returns:
        bool: True if the content is valid False otherwise.
    """
    for phrase in exclude_phrases:
        if phrase in content:
            return False
    return True

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
    
    pdf_path = config["lotus_tech_pdf"]
    output_file_name = config["output_tech"]
    output_directory = config["output_dir"]
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, output_file_name)
    
    start_page = int(config["tech_start"])
    end_page = int(config["tech_end"])

    exclude_phrases = [
        "Please Consider the Environment Before Printing This Document",
        "TABLE OF CONTENTS",
        "CONTENTS"
    ]

    data = read_specific_pages(pdf_path, start_page, end_page)

    output_data = []
    id_counter = 1
    for page_num, text in data:
        # For each page we split text by sentences
        chunks = split_text_by_sentences(text)
        for chunk in chunks:
            # Check if each chunk is valid else excluded
            if is_valid_content(chunk, exclude_phrases):
                output_data.append({
                    "page_number": page_num,
                    "content": chunk,
                    "id": id_counter
                })
                id_counter += 1
            else:
                print(f"Excluded content: {chunk}")

    save_json(output_data, output_file_path)

    print(f"Total valid chunks: {len(output_data)}")

if __name__ == "__main__":
    nltk.download('punkt')
    main()

