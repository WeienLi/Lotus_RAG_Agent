import os
import sys
import json
import logging
import yaml
import traceback
import time
import uuid
from functools import wraps
from flask import Flask, request, Response, stream_with_context, render_template, session
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama

from utils.chromaManager import ChromaManager
from utils.ollamaManager import OllamaManager
from utils.apiOllamaManager import ChatManager

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For Flask session
CORS(app)

api_chat_managers = {}  # Save different session_id as key for different ChatManager instance as value
langchain_chat_managers = {} # Save different session_id as key for different OllamaManager instance as value


def get_rag_content(response):
    """Extracts content from documents retrieved and reformat it

    Args:
        response (list): A list of document objects

    Returns:
        str: A formatted string containing extracted text follow by car stats.
    """
    rag_content = ""
    for i, doc in enumerate(response):
        page_content = doc.page_content.replace('\n', '')
        if len(page_content) < 50:
            continue
        car_stats = doc.metadata.get('car_stats', None)
        if car_stats:
            car_stat_content = ""
            for car_stat in car_stats:
                car_stat_content += car_stat.replace('\n', ' ')
            rag_content += f"{page_content}\nCar stats Related: {car_stat_content}\n"
        else:
            rag_content += f"{page_content}\n"

    return rag_content


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
    logging.info("Starting warm up")
    try:
        llm = ChatOllama(model=config['llm'])
        llm.invoke("Warm up")
        # sllm = ChatOllama(model=config['sllm'])
        # sllm.invoke("Warm up")
    except Exception as e:
        logging.error(f'Warm-up request failed: {str(e)}')


@app.route('/chat', methods=['POST'])
def api_chat():
    """
    Utilizing the Ollama API chat manager developed to handle chat requested. Use RAG if classified as needed. (apiOllamaManager)

    Returns:
        Response: A streamed response containing chat messages
    """
    try:
        data = request.json
        question = data.get('question')
        session_id = session.get('session_id1')

        if not question:
            return GlobalResponseHandler.error(message="Question not provided")

        if not session_id or session_id not in api_chat_managers:
            return GlobalResponseHandler.error(message="Session ID not found or Chat Manager not initialized")

        chat_manager = api_chat_managers[session_id]

        start_time = time.perf_counter()

        def generate_response():
            user_input = question
            use_rag = chat_manager.if_query_rag(user_input)
            rag_context = ""

            if use_rag == 'need rag':
                for retriever in retrievers:
                    rag_context += get_rag_content(retriever.invoke(user_input))
                    rag_context += '\n'

            response = chat_manager.chat(user_input, rag_context, True)

            first_response_time = time.perf_counter()
            response_time = first_response_time - start_time

            logging.info(f"Time to first response: {response_time:.2f} seconds")

            first_response_logged = False
            full_response = ""

            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode('utf-8'))
                        if 'message' in json_line and 'content' in json_line['message']:
                            content = json_line['message']['content']
                            full_response += content
                            json_data = json.dumps({'response': content, 'general_or_rag': use_rag})
                            yield f"data: {json_data}\n\n"
                            if not first_response_logged:
                                first_response_logged = True
                                logging.info(f"First token sent in {response_time:.2f} seconds")
                    except json.JSONDecodeError:
                        pass

            logging.info(f"Full response: {full_response}")
            logging.info(f"Flag: {use_rag}")

            chat_manager.save_chat_history(full_response)

        return Response(stream_with_context(generate_response()), content_type='text/event-stream')

    except Exception as e:
        logging.error(f"An error occurred in /chat endpoint: {str(e)}")
        logging.error("".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
        return GlobalResponseHandler.error(message=str(e))


@app.route('/langchain_chat', methods=['POST'])
def langchain_chat():
    """
     Utilizing the Langchain chat manager developed to handle chat requested. (ollamaManager)

    Returns:
        Response: A streamed response containing chat messages

    """
    try:
        data = request.json
        question = data.get('question')
        session_id = session.get('session_id2')

        if not question:
            return GlobalResponseHandler.error(message="Question not provided")

        if not session_id or session_id not in langchain_chat_managers:
            return GlobalResponseHandler.error(message="Session ID not found or Chat Manager not initialized")

        ollama_manager = langchain_chat_managers[session_id]

        start_time = time.perf_counter()

        def generate_response():
            gen = ollama_manager.chat(question, session_id)
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


@app.route('/test_api_chat')
def test_api_chat():
    """
     Initializes a test session for Ollama API chat manager and renders the test interface (apiOllamaManager)

    Returns:
        Response: Renders the `test_api.html` template for testing API chat
    """
    session_id = str(uuid.uuid4())
    session['session_id1'] = session_id
    api_chat_managers[session_id] = ChatManager(session_id, base_url, model_name)
    return render_template('test_api.html')


@app.route('/test_langchain_chat')
def test_langchain_chat():
    """
    Initializes a test session for LangChain chat manager and renders the test interface. (ollamaManager)
    
    Returns:
        Response: Renders the `test_langchain.html` template for testing LangChain chat.
    """
    session_id = str(uuid.uuid4())
    session['session_id2'] = session_id
    # Use retrievers[0] since we found that lotus the combined collections perform the best.
    langchain_chat_managers[session_id] = OllamaManager(config, retrievers[0])
    return render_template('test_langchain.html')


@app.route('/favicon.ico')
def favicon():
    """
    Handles requests for the favicon to prevent unnecessary 404 errors.
        
    Returns:
        Response: An empty response with HTTP status code 204 (No Content).
    """
    return '', 204


if __name__ == "__main__":
    config_path = os.getenv('CONFIG_PATH', './config/config.yaml')
    config = load_config(config_path)
    base_url = config.get('ollama_base_url')
    model_name = config.get('llm')
    if not model_name or not base_url:
        logging.error("LLM model name/base_url is not configured.")
        sys.exit(1)

    logging.info(f"Using model: {model_name}, URL: {base_url}")

    #Which collections we want to use and how much we want to extract (The current values set is the best we found by expriment)
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    #List of retrievers
    retrievers = []
    for collection, top_k in collections.items():
        if top_k <= 0:
            continue
        chroma_manager = ChromaManager(config=config, collection_name=collection)
        chroma_manager.create_collection()
        retrievers.append(chroma_manager.get_retriever(k=top_k, retriever_type="ensemble"))

    warm_up(config)

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 6006)))
