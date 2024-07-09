import fitz  # PyMuPDF
import json
import os

def extract_text_by_page(file_path):
    document = fitz.open(file_path)
    pages_content = []
    pages_to_omit = {1, 2, 4, 5, 7, 8}
    for page_num in range(len(document)):
        page_number = page_num + 1  # Page numbers start from 1
        if page_number in pages_to_omit:
            continue
        page = document.load_page(page_num)
        text = page.get_text()
        pages_content.append({"page_number": page_number, "content": text})

    return pages_content

def save_text_to_json(pages_content, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pages_content, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    file_path = "./Lotus The Marques.pdf"
    output_directory = "."
    output_filename = "by_page_splited2.json"
    output_path = os.path.join(output_directory, output_filename)

    os.makedirs(output_directory, exist_ok=True)

    pages_content = extract_text_by_page(file_path)
    print("Text extraction by page complete.")

    save_text_to_json(pages_content, output_path)
    print(f"Documents have been split by page and saved to {output_path}")
