from easyinstruct import ICLPrompt
from easyinstruct.utils.api import set_openai_key
import openai
from datasets import load_metric
from transformers import AutoTokenizer
import time


# 定义翻译任务
def translate(model, input_text, target_lang="chinese"):

    source_lang = "uyghur"
    prompts = f"Please translate {source_lang} sentences {input_text} into {target_lang}, let's think step by step "

    # Step1: Set your own API-KEY

    set_openai_key("XXXXXXXXXXX")

    # Step2: Declare a prompt class
    prompt = ICLPrompt()

    # Step3: Desgin a few task-specific examples
    in_context_examples = [{
        '''Please translate uyghur sentences "بۈگۈنكى بۇ ھەيۋەتلىك ھەم مۇقەددەس مۇراسىمغا قاتناشقان پايتەختتىكى ھەر ساھە ۋەكىللىرى قاتتىق ھاياجانلاندى." into chinese, let's think step by step''':
            '''
                Step 1: Identify the language of the original sentence.
                The original sentence is in uyghur.

                Step 2: Translate the sentence into English.
                Participating in today's solemn and sacred ceremony made the representatives of all walks of life in the capital city each feel a great deal of emotion.

                Step 3: Translate the sentence from English to Chinese.
                参加今天这个庄严神圣的仪式，让首都各界代表个个感慨万千。

             '''},

    ]

    # Step4: Build a prompt from the examples
    prompt.build_prompt(prompts, in_context_examples, n_shots=1)

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
    lines = []
    file = open(root, 'r', encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        lines.append(line)
    return lines


input_texts = read_file('./ccmt/2021uc.ug.txt')
labels_list = read_file('./ccmt/2021uc.zh.txt')


all_bleu = 0
f1 = open('pred/COT/ug-zh/cot-ug-zh-1shot.txt', 'a', encoding='utf-8')
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