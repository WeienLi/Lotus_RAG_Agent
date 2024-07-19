import pdfplumber
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

pdf_path = "D:/Lotus/20-F.pdf"
output_file_path = 'D:/Lotus/20-F.json'
start_page = 76
end_page = 123

def read_specific_pages(pdf_path, start_page, end_page):
    with pdfplumber.open(pdf_path) as pdf:
        specific_pages_data = []
        for page_num in range(start_page, end_page + 1):
            page = pdf.pages[page_num - 1]
            text = page.extract_text()
            specific_pages_data.append((page_num, text))
    return specific_pages_data

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

data = read_specific_pages(pdf_path, start_page, end_page)

output_data = []
id_counter = 1
for page_num, text in data:
    chunks = split_text_by_sentences(text)
    for chunk in chunks:
        output_data.append({
            "page_number": page_num,
            "content": chunk,
            "id": id_counter
        })
        id_counter += 1

with open(output_file_path, 'w') as file:
    json.dump(output_data, file, indent=4)
