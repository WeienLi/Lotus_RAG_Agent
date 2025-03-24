from sentence_transformers import SentenceTransformer, util
import json
import yaml
"""
RAG_chroma.py

This is a prototype testing file that is first used to test the RAG functionality without using vector database
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

def load_json(file):
    """load documents from JSON file

    Args:
        file (str): path towards the JSON file we want to load

    Returns:
        list[dict]: A list of dictionaries where each dictionary represents a document from the JSON file.
    """
    with open('car_stats.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data
    
def calculate_similarity(model, prompt, chunks):
    """Calculate the similarity of prompt with each of the chunks

    Args:
        model (Embedding Model): The function that we use to embedded the data
        prompt (str): User's questions
        chunks (list[str]) : A list of string where each string represents the content of a document from the JSON file.

    Returns:
        torch.Tensor: A 1D tensor containing cosine similarity scores between the prompt and each chunk
    """
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(prompt_embedding, chunk_embeddings)[0]
    return similarities

    
def main():
    config_path = "./prototype_reference/config/config.yaml"
    config = load_config(config_path)
    embeddings_model_name = config["embeddings_model_name"]
    model = SentenceTransformer(embeddings_model_name)
    data = load_json(config["data_path"])
    chunks = [entry['content'] for entry in data]
    prompt = "What was Colin Chapman’s significant achievement at the ‘Eight Clubs’ meeting at Silverstone in June 1950?"
    top_k = 5
    similarities = calculate_similarity(model,prompt, chunks)
    top_k_indices = similarities.topk(top_k).indices.tolist()
    top_k_similarities = similarities[top_k_indices].tolist()
    for idx, sim in zip(top_k_indices, top_k_similarities):
        chunk = chunks[idx]
        entry = data[idx]
        print(f"Similarity: {sim:.4f}")
        print(f"Chunk: {chunk}")
        print(f"Page number: {entry['page_number']}")
        if 'car_stats' in entry:
            print(f"Car stats: {entry['car_stats']}")
        print("\n---\n")

if __name__ == "__main__":
    main()