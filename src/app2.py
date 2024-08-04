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
from utils.apiOllamaManager import ChatManager

os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于Flask session
CORS(app)

chat_managers = {}  # 存储不同session_id对应的ChatManager实例


def get_rag_content(response):
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
    @staticmethod
    def success(data=None, message="Success", status_code=200, response_time=None):
        return GlobalResponseHandler._create_response("success", message, data, status_code, response_time)

    @staticmethod
    def error(message="An error occurred", data=None, status_code=400, response_time=None):
        return GlobalResponseHandler._create_response("error", message, data, status_code, response_time)

    @staticmethod
    def _create_response(status, message, data, status_code, response_time):
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
        return Response(stream_with_context(generate_func()), content_type='text/event-stream')


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def timing_decorator(func):
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
    logging.info("Starting warm up")
    try:
        llm = ChatOllama(model=config['llm'])
        llm.invoke("Warm up")
        # sllm = ChatOllama(model=config['sllm'])
        # sllm.invoke("Warm up")
    except Exception as e:
        logging.error(f'Warm-up request failed: {str(e)}')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    session_id = session.get('session_id')

    if not question:
        return GlobalResponseHandler.error(message="Question not provided")

    if not session_id or session_id not in chat_managers:
        return GlobalResponseHandler.error(message="Session ID not found or Chat Manager not initialized")

    chat_manager = chat_managers[session_id]

    try:
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


@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An unexpected error occurred: {str(e)}")
    return GlobalResponseHandler.error(message="Internal Server Error")


@app.route('/test_chat')
def test_chat():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    chat_managers[session_id] = ChatManager(session_id, base_url, model_name)
    return render_template('test_chat.html')


@app.route('/favicon.ico')
def favicon():
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

    # create retrievers
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    retrievers = []
    for collection, top_k in collections.items():
        if top_k <= 0:
            continue
        chroma_manager = ChromaManager(config=config, collection_name=collection)
        chroma_manager.create_collection()
        retrievers.append(chroma_manager.get_retriever(k=top_k, retriever_type="ensemble"))

    warm_up(config)

    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 6006)))
