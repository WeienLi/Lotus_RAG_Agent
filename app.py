import os
import sys
import subprocess
import time
import json
import logging
import yaml
import requests
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from db.chromaManager import ChromaManager
from src.ollamaManager import OllamaManager

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', 'What is Mark')

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    results = chroma_manager.retrieve_top_k(question)
    context = "\n\n".join([result[0].page_content for result in results])

    def generate_response():
        for partial_response in ollama_manager.chat(context, question):
            json_data = json.dumps({'response': partial_response})
            yield f"data: {json_data}\n\n"
            time.sleep(0.01)

    return Response(stream_with_context(generate_response()), content_type='text/event-stream')


if __name__ == "__main__":
    # check_and_start_service(model_name)
    config_path = "./config/config.yaml"
    config = load_config(config_path)

    model_name = config['llm']

    chroma_manager = ChromaManager(config, 'lotus')
    chroma_manager.load_model()

    ollama_manager = OllamaManager(model=model_name)

    app.run(host='0.0.0.0', port=5001, debug=True)
