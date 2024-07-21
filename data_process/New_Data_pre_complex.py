import pdfplumber
import json
import nltk
import re

#pdf_path = "D:/Lotus/20-F.pdf"
#output_file_path = 'D:/Lotus/20-F.json'
#start_page = 76
#end_page = 123

pdf_path = "D:/Lotus/LotusTech-F-4A_2023_12_05.pdf"
output_file_path = 'D:/Lotus/LotusTech-F-4A_2023_12_05.json'
start_page = 275
end_page = 352

exclude_phrases = [
    #"Please Consider the Environment Before Printing This Document",
    #"TABLE OF CONTENTS",
    #"CONTENTS"
]

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def read_specific_pages(pdf_path, start_page, end_page):
    with pdfplumber.open(pdf_path) as pdf:
        specific_pages_data = []
        for page_num in range(start_page, end_page + 1):
            page = pdf.pages[page_num - 1]
            text = page.extract_text()
            cleaned_text = clean_text(text)
            specific_pages_data.append((page_num, cleaned_text))
    return specific_pages_data

def clean_text(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not is_header_or_footer(line)]
    cleaned_text = ' '.join(filtered_lines)
    return cleaned_text

def is_header_or_footer(line):
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
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_word_count + sentence_word_count > max_word_count:
            if current_word_count >= min_word_count:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def is_valid_content(content, exclude_phrases):
    for phrase in exclude_phrases:
        if phrase in content:
            return False
    return True

data = read_specific_pages(pdf_path, start_page, end_page)

output_data = []
id_counter = 1
for page_num, text in data:
    chunks = split_text_by_sentences(text)
    for chunk in chunks:
        if is_valid_content(chunk, exclude_phrases):
            output_data.append({
                "page_number": page_num,
                "content": chunk,
                "id": id_counter
            })
            id_counter += 1
        else:
            print(f"Excluded content: {chunk}")

with open(output_file_path, 'w') as file:
    json.dump(output_data, file, indent=4)

print(f"Total valid chunks: {len(output_data)}")

