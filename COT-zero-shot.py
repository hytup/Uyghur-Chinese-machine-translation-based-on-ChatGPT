from easyinstruct import BasePrompt
from easyinstruct.utils.api import set_openai_key
import openai
from datasets import load_metric
from transformers import AutoTokenizer
import time

# 定义翻译任务
def translate(model , input_text , target_lang="chinese"):
    # input_text = "Today is a good day."
    source_lang = "uyghur"

    prompts =  f"Please translate {source_lang} sentences {input_text} into {target_lang}, let's think step by step "

    set_openai_key("XXXXXXXXXXXXX")

    # Step2: Declare a prompt class
    prompt = BasePrompt()

    # Step3: Build a prompt
    prompt.build_prompt(prompts)

    # Step4: Get the result from LLM API service
    translation = prompt.get_openai_result(engine=model)


    # if model == 'gpt-3.5-turbo':
    #
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",
    #         messages=[
    #             {"role": "system", "content": "You are a chinese translator."},
    #             {"role": "user", "content": prompt}]
    #     )
    #     translation = response.choices[0].message.content.strip()
    # else:
    #     response = openai.Completion.create(
    #         engine=model,
    #         prompt=prompt,
    #         max_tokens=200,
    #         n=1,
    #         stop=None,
    #         temperature=0,
    #     )
    #     translation = response.choices[0].text.strip()
    return translation

def read_file(root):
    lines =[]
    file = open(root,'r',encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        lines.append(line)
    return lines[0:100]


input_texts = read_file('./ccmt/2021uc.ug.txt')
labels_list = read_file('./ccmt/2021uc.zh.txt')



all_bleu=0
f1 =open('pred/COT/ug-zh/cot-ug-zh.txt', 'a', encoding='utf-8')
for labels, input_text in zip(labels_list, input_texts):
    # 调用翻译函数
    while True:
        try:
            translations = translate('gpt-3.5-turbo', input_text)
            break
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting for 20 seconds...")
            time.sleep(20)

    # 打印翻译结果
    f1.write(translations)
    f1.write('\n')
    print(f"Translation: {translations}")

f1.close()

