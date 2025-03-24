# Lotus RAG Agent

## Overview:
**RAG Agent for Lotus** is a **Retrieval-Augmented Generation (RAG) system** designed to process and enhance responses with contextually relevant information from a knowledge base. The system integrates **ChromaDB** for retrieval and **Ollama** for LLM-based response generation, optimizing information extraction related to Lotus vehicles and Lotus brand informations.

## Features:

- **Data Extraction** Extracts from PDF file and store them in a JSON format, different data have different extracting and chunking strategy. 
- **Data Refinment** Data Refinement Utilize KIMI AI API to refine the chunks to enhance the retrieval accuracy.
- **Vector Database Integration** Utilizes ChromaDB for stores and retrieves indexed knowledge for efficient querying with a custom manager.
- **API RAG Framework** Directly utilizes the Ollama API for the entire RAG process. (Determines RAG or not) -> (Do simple RAG else directly generate)
- **Langchain RAG Framework** Utilizes Langchain for the entire RAG process. (Obtains Model Internal knowledge) -> (Pass in the internal knowledge alongside the data retrieved) -> (Final Generation)
- **Flask API** Both Framework implement their own Flask API for interaction.
- **Complete Testing Framework** Both Framework can be tested upon using our test either by interactive shell or a list of queries. Furthermore it can also evaluates our retrieval accuracy and RAG or not Flag accuracy.
- **Prototype Reference** Simple Prototype that can be easily implemented and play around

## Installation:

To install our Lotus RAG Agent project first clone the repository

```
git clone https://github.com/WeienLi/Lotus_RAG_Agent.git
```

Then install the necessary dependencies:

```
cd path/to/Lotus_RAG_Agent
pip install -r requirements.txt
```

Keep requirements.txt up to date if you install some new packages:

```
pip freeze > requirements.txt
```

## Setup:

To set up the agent first update the config file and upload the necessary data to the ./data folder then execute the following to first preprocess the data. 

```
chmod u+x ./data_process/data_process.sh
./data_process/data_process.sh
```

**Optional**: 
If you would like to refine the data using KIMI AI make sure to update the refine_process.sh to input your API key. Then execute the following: 

```
chmod u+x ./data_process/refine_process.sh
./data_process/refine_process.sh
```

To organize the files and setup the database run the following:

```
chmod u+x ./data/org_and_loaddb.sh
/data/org_and_loaddb.sh
```

## Starting the service:

To start the service you can choose between **src/app2.py** and **src/app.py**. Where **src/app.py** utilizes the langchain RAG framework for more details see **src/utils/ollamaManager.py**. **src/app2.py** on the other hand utilizes the API RAG framework for more details see **src/utils/apiOllamaManager.py**.

###  Key Differences Between the Two Frameworks

| Feature              | API RAG Framework                          | Langchain RAG Framework                    |
|----------------------|------------------------------------------|--------------------------------------------|
| **Retrieval Logic**  | Uses the Large Language Model to determine if RAG is needed | Always retrieves both internal and external knowledge |
| **Generation Process** | Performs simple retrieval if needed, otherwise directly generates a response | Passes both retrieved knowledge and model insights for final generation |
| **Integration**      | Lightweight, relies on external API calls | Uses Langchain to integrate retrieval with LLM generation |
| **Use Case**        | Best for easier task with direct responses and minimal processes | Best for complex question where it requires deep retrieval and context-rich answers |

```
python app.py
```

or 

```
python app2.py
```

## Tests: 
For test we have a complete test suite to test the performance of our service please consult **./src/test** for more information and choose the test you would like to perform. 

```
python ./src/test/test_you_want.py
```