import requests
import json
import sys

def stream_chat(prompt, url):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "question": prompt
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        print(f"Response content: {response.text}")
        return None

    #sys.stdout.write("Response from model: ")
    #sys.stdout.flush()

    response_text = "" 
    for line in response.iter_lines(): 
        if line: decoded_line = line.decode('utf-8') 
        if decoded_line.startswith('data:'): data = json.loads(decoded_line.split('data: ')[-1]) 
        if 'done' in data and data['done']: break 
        response_part = data.get('response', '') 
        if response_part:
            readable_response = response_part.encode('utf-8').decode('unicode_escape') 
            response_text += readable_response 
            sys.stdout.write(readable_response) 
            sys.stdout.flush()
    print("\n")
    return response_text

def test_chat(url, questions):
    results = []

    for question in questions:
        print(f"Question: {question}")
        response_text = stream_chat(question, url)
        
        if response_text is not None:
            results.append({
                "question": question,
                "response": response_text
            })

    return results

if __name__ == "__main__":
    url = "http://127.0.0.1:6006/chat"
    

    questions = [
        
    ]


    results = test_chat(url, questions)

    with open('question_test.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)