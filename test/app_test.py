import requests
import json
import sys


def stream_chat(prompt, url):
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


def test_chat(url):
    while True:
        prompt = input("Enter your question (or exit): ")
        if prompt.lower() == "exit":
            break

        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "question": f"{prompt}"
        }

        response = requests.post(url, headers=headers, json=payload, stream=True)

        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            return

        # sys.stdout.write("Response from model: ")
        sys.stdout.flush()

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
                            response_text = data.get('response', '')
                            flag = data.get('general_or_rag', '')
                            # print(flag)
                            if response_text:
                                sys.stdout.write(response_text)
                                # sys.stdout.write(f"Response: {response_text}\n")
                                # sys.stdout.write(f"general_or_rag: {flag}\n")
                                sys.stdout.flush()
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            buffer = ""
        # print(flag)
        print("\n")


if __name__ == "__main__":
    # user_input = input("Enter your prompt: ")
    # user_input = "What modifications were made to the Seven's chassis and rear extensions, and who assisted Colin in building the Mark IIIs?"
    # stream_chat(user_input)
    url = "http://127.0.0.1:6006/chat"
    test_chat(url)
