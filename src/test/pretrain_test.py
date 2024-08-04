import subprocess
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def run_ollama(question, config):
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

    results = test_ollama_run(config, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/q3.txt', 'w') as outfile:
        for result in results:
            outfile.write(f"Question: {result['question']}\n")
            outfile.write(f"Response: {result['response']}\n")
            outfile.write("=" * 50 + "\n")
