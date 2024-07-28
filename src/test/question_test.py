import requests
import json
import sys


def stream_chat(prompt, url):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "question": prompt
    }

    response = requests.post(url, headers=headers, json=payload, stream=True)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        print(f"Response content: {response.text}")
        return None

    response_text = ""
    buffer = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                buffer += decoded_line.split('data: ')[-1].strip()
                if buffer.endswith("}"):
                    try:
                        data = json.loads(buffer)
                        buffer = ""
                        if 'done' in data and data['done']:
                            break
                        response_part = data.get('response', '')
                        if response_part:
                            response_text += response_part
                            sys.stdout.write(response_part)
                            sys.stdout.flush()
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        buffer = ""

    print("\n")
    return response_text


def test_chat(url, questions):
    results = []

    for question in questions:
        print(f"Question: {question}")
        response_text = stream_chat(question, url)

        if response_text is not None:
            results.append({
                "question": question,
                "response": response_text
            })

    return results


if __name__ == "__main__":
    url = "http://127.0.0.1:6006/chat"

    questions = [
        "1.Lotus在F1历史上有哪些值得铭记的成就？能否分享一些具体的故事或数据？",
        "2.Lotus的哪些技术创新对整个汽车行业产生了深远影响？",
        "3.在设计Lotus跑车时，您是如何平衡美学和空气动力学的？",
        "4.能否详细介绍一下Lotus的轻量化设计理念及其在实际车型中的应用？",
        "5.Lotus的哪些车型在赛道上的表现最为出色？它们有哪些独特的技术或设计特点？",
        "6.与法拉利、兰博基尼等其他豪华跑车品牌相比，Lotus的竞争优势在哪里？",
        "7.Lotus在环保和可持续发展方面有哪些举措或计划？",
        "8.您能分享一下Lotus在电动跑车领域的最新进展吗？",
        "9.Lotus的哪些车型是为赛道专门设计的？它们有哪些专为赛道优化的特性？",
        "10.Lotus的客户定制服务包括哪些内容？客户可以进行哪些个性化选择？",
        "11.在Lotus的历史上，有哪些车型是限量版或特别版？它们有什么独特之处？",
        "12.Lotus在车辆安全技术方面有哪些创新？",
        "13.能否介绍一下Lotus的悬挂系统是如何提升驾驶体验的？",
        "14.Lotus的哪些车型在历史上被认为是设计或性能的标杆？",
        "15.作为潜在车主，我如何能够体验Lotus的赛道日活动？",
        "16.Lotus在车辆维护和售后服务方面提供哪些支持？",
        "17.您能分享一些Lotus车主社区的有趣故事或活动吗？",
        "18.Lotus在赛车模拟器或电子游戏中有哪些合作项目？这些项目对品牌有何影响？",
        "19.您如何看待Lotus在当前汽车市场中的定位？它如何应对日益激烈的竞争？",
        "20.对于像我这样对F1和赛车文化有深入了解的车主，Lotus有哪些特别的推荐或建议？",
        "21.您能简单介绍一下Lotus品牌的历史和它在汽车行业中的地位吗？",
        "22.Lotus有哪些主要的车型？它们各自的特点是什么？",
        "23.您能详细解释一下Lotus在赛车领域的成就吗？",
        "24.我听说Lotus的跑车很轻，这在驾驶体验上有什么特别之处吗？",
        "25.我预算在80-150万港币，Lotus有哪些车型适合我？",
        "26.相比于保时捷、奔驰和玛莎拉蒂，Lotus的性价比如何？",
        "27.Lotus的跑车在安全性方面有哪些特别的设计或技术？",
        "28.您能介绍一下Lotus的售后服务和保养服务吗？",
        "29.我家里的BMW 5系轿车是四门的，Lotus的跑车是否适合家庭使用？",
        "30.购买Lotus的跑车，我需要考虑哪些额外的费用？",
        "31.Lotus的跑车在二手市场上的保值率如何？",
        "32.您能分享一下Lotus车主的用车体验吗？",
        "33.Lotus的哪些车型在驾驶乐趣方面特别突出？",
        "34.我听说Lotus的跑车设计很独特，您能具体介绍一下吗？",
        "35.我不太了解赛车，Lotus的跑车在日常驾驶中的表现如何？",
        "36.购买Lotus的跑车，我可以获得哪些定制服务？",
        "37.Lotus的跑车在燃油效率方面表现如何？",
        "38.您能介绍一下Lotus的电动跑车吗？它们与传统燃油车有什么不同？",
        "39.我计划在周末偶尔驾驶跑车外出，Lotus的跑车是否适合长途驾驶？",
        "40.您能推荐一些适合初学者的Lotus车型吗？"

    ]

    results = test_chat(url, questions)

    with open('/root/autodl-tmp/RAG_Agent/data/test_data/qwen2:72b_result.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
