def decode_unicode_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            content = input_file.read()

        readable_content = content.encode('utf-8').decode('unicode_escape')

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(readable_content)

        print(f"Decoded content written to {output_file_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_file = "/root/autodl-tmp/RAG_Agent/data/test_data/llama3:8b_result.json"
    output_file = "/root/autodl-tmp/RAG_Agent/data/test_data/llama3:8b_result40.json"
    
    decode_unicode_file(input_file, output_file)