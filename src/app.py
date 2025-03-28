import os
import sys
import json
import logging
import yaml
import traceback
import time
from functools import wraps
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama

from utils.chromaManager import ChromaManager
from utils.ollamaManager import OllamaManager

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


class GlobalResponseHandler:
    """
    A global reponse handler for formatting API responses
    """
    @staticmethod
    def success(data=None, message="Success", status_code=200, response_time=None):
        """Returns a success response

        Args:
            data (_type_, optional): Response Data. Defaults to None
            message (str, optional): Response Message. Defaults to "Success"
            status_code (int, optional): HTTP status code. Defaults to 200
            response_time (_type_, optional): Time it took for the response. Defaults to None

        Returns:
           Response: A Flask Response object containing the JSON response
        """
        return GlobalResponseHandler._create_response("success", message, data, status_code, response_time)

    @staticmethod
    def error(message="An error occurred", data=None, status_code=400, response_time=None):
        """Returns a error response

        Args:
            message (str, optional): Error Message. Defaults to "An error occurred"
            data (_type_, optional): Response Data. Defaults to None
            status_code (int, optional): HTTP status code. Defaults to 400
            response_time (_type_, optional): Time it took for the response. Defaults to None

        Returns:
            Response: A Flask Response object containing the JSON response
        """
        return GlobalResponseHandler._create_response("error", message, data, status_code, response_time)

    @staticmethod
    def _create_response(status, message, data, status_code, response_time):
        """Helper function that creates a JSON response object.

        Args:
            status (str): The status of the response
            message (str): The response message
            data (any): The response data
            status_code (int): The HTTP status code
            response_time (float, optional): The time taken to generate the response 

        Returns:
            Response: A Flask Response object containing the JSON response
        """
        response = {
            "status": status,
            "message": message,
            "data": data,
            "response_time": response_time
        }
        response_json = json.dumps(response)
        return Response(response=response_json, status=status_code, mimetype='application/json')

    @staticmethod
    def stream_response(generate_func):
        """Streams a response using the provided generator function

        Args:
            generate_func (function): A generator function that yields response data

        Returns:
            Response: A Flask Response object with a streamed response
        """
        return Response(stream_with_context(generate_func()), content_type='text/event-stream')


def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def timing_decorator(func):
    """A decorator that logs the execution time of the wrapped function

    Args:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function with timing logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds")
        return result

    return wrapper


@timing_decorator
def warm_up(config):
    """Warms up the language model by invoking it with a sample input

    Args:
        config (dict): The configuration dictionary containing model settings
    """
    try:
        llm = ChatOllama(model=config['llm'])
        llm.invoke("Warm up")
        # sllm = ChatOllama(model=config['sllm'])
        # sllm.invoke("Warm up")
    except Exception as e:
        logging.error(f'Warm-up request failed: {str(e)}')


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests by streaming responses from the OllamaManager model

    Returns:
        Response: A Flask streamed JSON response object containing the model's replies
    """
    data = request.json
    question = data.get('question')

    if not question:
        return GlobalResponseHandler.error(message="Question not provided")

    try:
        start_time = time.perf_counter()

        def generate_response():
            gen = ollama_manager.chat(question, 'ab123')
            first_response_time = time.perf_counter()
            response_time = first_response_time - start_time

            logging.info(f"Time to first response: {response_time:.2f} seconds")

            first_response_logged = False
            full_response = ""
            rag_or_general = "unknown"

            for item in gen:
                if isinstance(item, tuple) and item[0] == "FLAG":
                    rag_or_general = item[1]
                else:
                    full_response += item
                    json_data = json.dumps({'response': item, 'general_or_rag': rag_or_general})
                    yield f"data: {json_data}\n\n"
                    if not first_response_logged:
                        first_response_logged = True
                        logging.info(f"First token sent in {response_time:.2f} seconds")

            logging.info(f"Full response: {full_response}")
            logging.info(f"Flag: {rag_or_general}")

        return Response(stream_with_context(generate_response()), content_type='text/event-stream')

    except Exception as e:
        logging.error(f"An error occurred in /chat endpoint: {str(e)}")
        logging.error("".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
        return GlobalResponseHandler.error(message=str(e))



@app.errorhandler(Exception)
def handle_exception(e):
    """
    Handles unexpected exceptions and returns a generic error response

    Args:
        e (Exception): The exception that occurred

    Returns:
        Response: A Flask Response object for standarized error response
    """
    logging.error(f"An unexpected error occurred: {str(e)}")
    return GlobalResponseHandler.error(message="Internal Server Error")


if __name__ == "__main__":
    config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
    config = load_config(config_path)

    model_name = config.get('llm')
    #smodel_name = config.get('sllm')
    if not model_name:
        logging.error("LLM model name is not configured.")
        sys.exit(1)

    logging.info(f"Using model: {model_name}")
    #logging.info(f"Streaming model: {smodel_name}")

    #try:
    chroma_manager = ChromaManager(config, 'lotus')
    chroma_manager.create_collection()
    db_ret = chroma_manager.get_retriever(k=5, retriever_type="ensemble")
    ollama_manager = OllamaManager(config, db_ret)

    warm_up(config)

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 6006)))

    # except Exception as e:
    #     logging.error(f"An error occurred during initialization: {str(e)}")
    #     sys.exit(1)
