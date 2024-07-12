import requests
import json
import sys


def stream_chat(prompt):
    url = "http://localhost:5001/chat"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return

    sys.stdout.write("Response from model: ")
    sys.stdout.flush()

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                data = json.loads(decoded_line.split('data: ')[-1])
                if 'done' in data and data['done']:
                    break
                response_text = data.get('response', '')
                if response_text:
                    sys.stdout.write(response_text)
                    sys.stdout.flush()
    print("\n")


if __name__ == "__main__":
    # user_input = input("Enter your prompt: ")
    user_input = "What modifications were made to the Seven's chassis and rear extensions, and who assisted Colin in building the Mark IIIs?"
    stream_chat(user_input)
