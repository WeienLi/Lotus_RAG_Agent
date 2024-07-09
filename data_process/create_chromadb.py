import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
import chromadb
import json
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Define file path and collection name
file_path = "Data/split_car_stats.json"
collection_name = "test"

# Load the JSON file
loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
documents = loader.load()

# Initialize the SentenceTransformer model for embeddings
model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize persistent Chroma client
persist_directory = "./database"

# Initialize persistent Chroma client
persistent_client = chromadb.PersistentClient(path=persist_directory)
collection = persistent_client.get_or_create_collection(collection_name)

# Define Chroma instance with the correct persist directory
langchain_chroma = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=model,
    persist_directory=persist_directory
)
print(f"Persist directory: {persist_directory}")

content_list = []
metadata_list = []

for doc in documents:
    # Parse the content string to extract the "content" field
    content_dict = json.loads(doc.page_content)
    content = content_dict.get("content", "")
    content_list.append(content)
    metadata = {
        "page_number": content_dict.get("page_number"),
        "car_stats": json.dumps(content_dict.get("car_stats", "N/A")) if isinstance(content_dict.get("car_stats"),
                                                                                    list) else content_dict.get(
            "car_stats", "N/A"),
    }
    # print(content[:100])
    # print(metadata)
    metadata_list.append(metadata)
    # break

# Generate embeddings for the content
embeddings_list = model.embed_documents(content_list)
# Add documents with embeddings to Chroma
for content, embedding, metadata in zip(content_list, embeddings_list, metadata_list):
    langchain_chroma.add_texts([content], embeddings=[embedding], metadatas=[metadata])

# Save the Chroma database to the specified directory
langchain_chroma.persist()

print("Chroma vector database created and persisted successfully.")
