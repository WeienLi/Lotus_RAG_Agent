import os
import sys
import json
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apiOllamaManager import ChatManager
from utils.chromaManager import ChromaManager


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


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def test_chat_manager(config, session_id, retrievers, questions):
    base_url = config['ollama_base_url']
    model_name = config['llm']

    chat_manager = ChatManager(session_id, base_url, model_name)

    print(f"Start to chat with {model_name}.")

    results = []

    for user_input in questions:
        print(f"You: {user_input}")

        # use_rag = chat_manager.if_query_rag(user_input)
        use_rag = 'need rag'
        if use_rag == 'need rag':
            rag_context = ""
            for retriever in retrievers:
                rag_context += get_rag_content(retriever.invoke(user_input))
                rag_context += '\n'
        else:
            rag_context = ""
        response = chat_manager.chat(user_input, rag_context, True)
        ai_response = ""
        print(f"{use_rag} AI response: ", end="")
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    if 'message' in json_line and 'content' in json_line['message']:
                        content = json_line['message']['content']
                        ai_response += content
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print()

        chat_manager.save_chat_history(ai_response)
        print("=" * 50)

        results.append({
            "question": user_input,
            "response": ai_response
        })

    return results


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    config = load_config(config_path)
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    retrievers = []
    for collection, top_k in collections.items():
        if top_k <= 0:
            continue
        chroma_manager = ChromaManager(config=config, collection_name=collection)
        chroma_manager.create_collection()
        retrievers.append(chroma_manager.get_retriever(k=top_k, retriever_type="ensemble"))

 
    questions = [
        "1. What are the memorable achievements of Lotus in the history of F1? Can you share some specific stories or statistics?" ,
        "2. What technological innovations from Lotus have had a profound impact on the automotive industry as a whole?" ,
        "3. How did you balance aesthetics and aerodynamics when designing the Lotus sports car?" ,
        "4. Can you describe in detail the lightweight design concept of Lotus and its application in the actual vehicle?" ,
        "5. Which Lotus models perform best on the track? What are their unique technical or design features?" ,
        "6. What are Lotus' competitive advantages compared to other luxury sports car brands such as Ferrari and Lamborghini?" ,
        "7. What initiatives or plans does Lotus have in terms of environmental protection and sustainability?" ,
        "8. Can you share the latest progress of Lotus in the field of electric sports cars?" ,
        "9. Which Lotus models are specifically designed for the track? What features do they have that are optimized for the track?" ,
        "10. What does Lotus' customer customization service include? What personalization options are available to customers?" ,
        "11. In the history of Lotus, which models have been limited edition or special edition? What makes them unique?" ,
        "12. What are Lotus's innovations in vehicle safety technology?" ,
        "13. Can you explain how Lotus suspension improves the driving experience?" ,
        "14. Which Lotus models have historically been considered benchmarks of design or performance?" ,
        "15. As a potential car owner, how can I experience Lotus track Day events?" ,
        "16. What support does Lotus provide in terms of vehicle maintenance and after-sales service?" ,
        "17. Can you share some interesting stories or activities of the Lotus owner community?" ,
        "18. What are Lotus's collaborative projects in racing simulators or video games? How do these projects impact the brand?" ,
        "19. How do you see Lotus positioned in the current automotive market? How does it respond to increased competition?" ,
        "20. What special recommendations or recommendations does Lotus have for owners like me who have a deep understanding of F1 and racing culture?" ,
        "21. Can you briefly tell us the history of the Lotus brand and its position in the automotive industry?" ,
        "22. What are the main models of Lotus? What are their respective characteristics?" ,
        "23. Can you explain in detail the achievements of Lotus in the field of racing?" ,
        "24. I heard that the Lotus sports car is very light. What makes it special in terms of driving experience?" ,
        "25. I have a budget of HK $800,000 to HK $1.5 million. Which Lotus models are suitable for me?" ,
        "26. How does Lotus compare to Porsche, Mercedes and Maserati?" ,
        "27. What is the special design or technology of Lotus sports cars in terms of safety?" ,
        "28. Can you tell us about the Lotus after-sales service and maintenance service?" ,
        "29. I have a four-door BMW 5 Series at home. Is a Lotus sports car suitable for family use?" ,
        "30. What additional costs do I need to consider when purchasing a Lotus sports car?" ,
        "31. What is the retention rate of Lotus sports cars in the second-hand market?" ,
        "32. Can you share the experience of Lotus owners?" ,
        "33. Which Lotus models stand out in terms of driving pleasure?" ,
        "34. I heard that Lotus has a unique sports car design. Could you tell me more about it?" ,
        "35. I don't know much about racing. How does the Lotus car perform in everyday driving?" ,
        "36. What customizations can I get when I buy a Lotus sports car?" ,
        "37. How do Lotus sports cars perform in terms of fuel efficiency?" ,
        "38. Can you introduce the Lotus electric sports car? How are they different from conventional fuel vehicles?" ,
        "39. I plan to drive the sports car occasionally on weekends. Is the Lotus sports car suitable for long drives?" ,
        "40. Can you recommend some Lotus models for beginners?"
    ]

    results = test_chat_manager(config, "test_session", retrievers, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/q1.txt', 'w') as outfile:
        for result in results:
            outfile.write(f"Question: {result['question']}\n")
            outfile.write(f"Response: {result['response']}\n")
            outfile.write("=" * 50 + "\n")
