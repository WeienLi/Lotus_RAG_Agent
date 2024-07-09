from sentence_transformers import SentenceTransformer, util
import json

# Load the model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Function to calculate similarity
def calculate_similarity(prompt, chunks):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(prompt_embedding, chunk_embeddings)[0]
    return similarities

# Load the JSON database
with open('car_stats.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare the text chunks
chunks = [entry['content'] for entry in data]

# Define the prompt
prompt = "What was Colin Chapman’s significant achievement at the ‘Eight Clubs’ meeting at Silverstone in June 1950?"

# Calculate similarities
top_k = 5

# Calculate similarities
similarities = calculate_similarity(prompt, chunks)

# Get the top k most similar chunks
top_k_indices = similarities.topk(top_k).indices.tolist()
top_k_similarities = similarities[top_k_indices].tolist()

# Print the top k most similar chunks with their metadata
for idx, sim in zip(top_k_indices, top_k_similarities):
    chunk = chunks[idx]
    entry = data[idx]
    print(f"Similarity: {sim:.4f}")
    #print(f"Chunk: {chunk}")
    print(f"Page number: {entry['page_number']}")
    if 'car_stats' in entry:
        print(f"Car stats: {entry['car_stats']}")
    print("\n---\n")