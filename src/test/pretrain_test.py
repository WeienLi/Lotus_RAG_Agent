import subprocess
import yaml

"""
This file is to test upon the capability of LLM to answer domain specific quesiton without RAG.
"""

def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def run_ollama(question, config):
    """
    Run an Ollama model to generate a response for a given question

    Args:
        question (str): The input query to be answered by the Ollama model
        config (dict): Configuration dictionary (currently not used)

    Returns:
        str or None: The generated response from the Ollama model if successful, otherwise `None`
    """
    #model_name = config['llm']
    #base_url = config['ollama_base_url']
    command = f"ollama run llama3.1:70b '{question}'"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error running ollama: {stderr.decode('utf-8')}")
        return None

    return stdout.decode('utf-8')


def test_ollama_run(config, questions):
    """_summary_

    Args:
        config (dict): Configuration dictionary
        questions (list[str]): A list of question we want to test querying the model

    Returns:
        list[dict]: A list of dictionary containing the question and its response from the Ollama model
    """
    results = []

    for question in questions:
        print(f"Question: {question}")

        response = run_ollama(question, config)
        if response:
            results.append({
                "question": question,
                "response": response
            })
            print(f"Response: {response}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    config = load_config(config_path)

    questions = [
        "Sensitive INFO"
    ]

    results = test_ollama_run(config, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/q3.txt', 'w') as outfile:
        for result in results:
            outfile.write(f"Question: {result['question']}\n")
            outfile.write(f"Response: {result['response']}\n")
            outfile.write("=" * 50 + "\n")
