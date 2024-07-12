from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import subprocess
import time
import logging
import sys
import os
from db.chromaManager import ChromaManager, load_config

app = Flask(__name__)
CORS(app)

OLLAMA_SERVER_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = "/root/autodl-tmp/LLM4Lotus/code/RAG_Agent/config/config.yaml"
config = load_config(config_path)
chroma_manager = ChromaManager(config, 'lotus')
chroma_manager.load_model()


def check_and_start_service():
    try:
        response = requests.post(OLLAMA_SERVER_URL, json={"model": MODEL_NAME, "prompt": "", "stream": False})
        if response.status_code == 200:
            logger.info("llama3-8b service is already running.")
            return
    except requests.exceptions.RequestException:
        logger.info("llama3-8b service is not running, attempting to start...")

    subprocess.Popen(['nohup', 'ollama', 'serve', '&'])
    logger.info("Starting llama3-8b service...")

    time.sleep(10)

    try:
        response = requests.post(OLLAMA_SERVER_URL, json={"model": MODEL_NAME, "prompt": "", "stream": False})
        if response.status_code == 200:
            logger.info("llama3-8b service has successfully started.")
        else:
            raise Exception("Failed to start llama3-8b service.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to start llama3-8b service: {e}")


def warm_up():
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": "Hello!",
        "stream": True
    }

    logger.info("Warming up the model...")
    logger.info(f"Sending request to {OLLAMA_SERVER_URL} with payload: {json.dumps(payload)}")
    response = requests.post(OLLAMA_SERVER_URL, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        raise Exception(f"Warm-up request failed with status code {response.status_code}")

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            logger.info(f"Warm-up response from model: {decoded_line}")
            break

    logger.info("Warm-up complete. Model is ready to use.")


@app.route('/chat', methods=['POST'])
def chat():
    logger.info("Received a request...")
    try:
        data = request.json
        logger.info(f"Request data: {data}")
        prompt = data.get('prompt')
        if not prompt:
            logger.error("No prompt provided in the request.")
            return jsonify({"error": "No prompt provided"}), 400

        results = chroma_manager.retrieve_top_k(prompt, k=3)

        # template
        context = "\n\n".join([result[0].page_content for result in results])

        combined_prompt = f"Context: {context}\n\nUser Question: {prompt}"
        # combined_prompt = f"{prompt}"
        print(combined_prompt)

        return Response(stream_request(combined_prompt), content_type='text/event-stream')

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


def stream_request(prompt):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(OLLAMA_SERVER_URL, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        yield f"data: {{\"error\": \"Request failed with status code {response.status_code}\"}}\n\n"
        return

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            logger.info(f"Response from model: {decoded_line}")
            yield f"data: {decoded_line}\n\n"


if __name__ == '__main__':
    # check_and_start_service()
    # warm_up()
    app.run(host='0.0.0.0', port=5001, debug=True)
