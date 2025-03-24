import requests
import json
import sys


def stream_chat(prompt, url):
    """
    Streams the chat response from an API in real-time

    Args:
        prompt (str): The query to be sent to the server
        url (str): The endpoint URL for the chat API

    Returns:
        str or None: The response text from the API if successful, otherwise `None`

    """
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

    response_text = ""
    buffer = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                buffer += decoded_line.split('data: ')[-1].strip()
                if buffer.endswith("}"):
                    try:
                        data = json.loads(buffer)
                        buffer = ""
                        if 'done' in data and data['done']:
                            break
                        response_part = data.get('response', '')
                        if response_part:
                            response_text += response_part
                            sys.stdout.write(response_part)
                            sys.stdout.flush()
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        buffer = ""

    print("\n")
    return response_text


def test_chat(url, questions):
    """
    Test the our current running API for its RAG generation using questions

    Args:
        url (str): The endpoint URL for the chat API
        questions (list[str]): A list of input queries to be tested

    Returns:
        list[dict]: A list of dictionaries containing the input query and answer pair
    """
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
    url = "http://127.0.0.1:6006/chat" #Loopback URL both app and app2 can be run on this 6006 port

    questions = [
        "Sensitive INFO"
    ]

    results = test_chat(url, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/gemma2:9b_result.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
