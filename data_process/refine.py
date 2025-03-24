import json
import sys
import time
from openai import OpenAI
from tqdm import tqdm

def refine_content_with_kimi(content, api_key):
    """
    Refines text content using the Moonshot AI API to refine the content

    Args:
        content (str): The raw text content to be refined
        api_key (str): The API key required for authenticating with the Moonshot AI service

    Returns:
        str | None: The refined text without special characters in string
                    Returns None if an error occurs during the API call
    """
    
    client = OpenAI(
        api_key=api_key,
        base_url='https://api.moonshot.cn/v1',
    )
    
    prompt = f"""
    Refine the following text by:
    1. Removing inappropriate symbols and replacing them with spaces or correct punctuation.
    2. Correcting obvious typos. Insert necessary spaces to between words.
    3. Removing incomplete sentences at the beginning and end such as "Table of Contents" but not followed with a table of contents.
    4. Formatting car stats with labels and values clearly separated by colons.
    Output the refined text as a single continuous paragraph using only normal human-readable symbols. Do not use any newline characters (\\n, \\n, /n, //n, /n-, \\n-), slashes (/), backslashes (\\\\), or other special characters.

    Input:
    {content}

    Output:
    """
    
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI, proficient in Chinese and English conversations. You provide safe, helpful, and accurate answers, while refusing to answer any questions related to terrorism, racism, violence, etc. 'Moonshot AI' is a proprietary term and should not be translated into other languages."},
        {"role": "user", "content": prompt}
    ]
    
    while True:
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                temperature=0.3,
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if "rate_limit_reached_error" in error_message:
                print("Rate limit reached. Sleeping for 15 seconds...")
                time.sleep(15)
            elif "high risk" in error_message:
                print("High risk error detected. Sleeping for 1 minute...")
                time.sleep(60)
            else:
                print(f"Error: {e}")
                return None

def process_chunks(input_file, output_file, start, end, api_key):
    """
    Processes a subset of JSON data, refines the content using the Moonshot AI API then saves the cleaned data to an output file

    Args:
        input_file (str): Path to the JSON file containing the data.
        output_file (str): Path to save the refined JSON output.
        start (int): The starting index of JSON data
        end (int): The ending index of JSON data
        api_key (str): API key for authenticating with the Moonshot AI service.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    refined_chunks = []

    for item in tqdm(data[start:end], desc=f"Processing chunks {start} to {end}"):
        combined_content = f"Content: {item['content']}"
        if item.get('car_stats'):
            car_stats_part = f" Car Stats: {item['car_stats'][0]}"
        else:
            car_stats_part = ""
        combined_content += car_stats_part

        while True:
            refined_chunk = refine_content_with_kimi(combined_content.strip(), api_key)
            time.sleep(1)  # Add a 1-second delay to avoid hitting rate limits
            if refined_chunk:
                try:
                    if "Car Stats:" in refined_chunk:
                        refined_content, refined_car_stats = refined_chunk.split("Car Stats:")
                        refined_content = refined_content.replace("Content:", "").strip()
                        refined_car_stats = refined_car_stats.strip()
                    else:
                        refined_content = refined_chunk.replace("Content:", "").strip()
                        refined_car_stats = ""

                    refined_data = {
                        "content": refined_content,
                        "page_number": item["page_number"],
                        "id": item["id"]
                    }
                    if refined_car_stats:
                        refined_data["car_stats"] = [refined_car_stats]
                    refined_chunks.append(refined_data)
                    break  # Break out of the while loop once successful
                except Exception as e:
                    print(f"Error processing refined chunk: {e}")
                    print(f"Refined chunk: {refined_chunk}")
                    break  # Break out of the while loop on processing error
            else:
                print("No refined data returned from the API. Retrying...")
                time.sleep(10)  # Sleep for 10 seconds before retrying

    if refined_chunks:
        final_output = json.dumps(refined_chunks, ensure_ascii=False, indent=4)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"Saved refined data to {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    api_key = sys.argv[5]
    process_chunks(input_file, output_file, start, end, api_key)
