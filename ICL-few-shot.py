from easyinstruct import ICLPrompt
from easyinstruct.utils.api import set_openai_key
import openai
from datasets import load_metric
from transformers import AutoTokenizer


# 定义翻译任务
def translate(model , input_text , target_lang="chinese"):

    source_lang = "uyghur"

    prompts =  f"Translate {source_lang} sentence {input_text} into {target_lang}: "

    # Step1: Set your own API-KEY
    set_openai_key("XXXXXXXXXXX")

    # Step2: Declare a prompt class
    prompt = ICLPrompt()

    # Step3: Desgin a few task-specific examples
    in_context_examples = [{
                               '''Translate uyghur sentence "بۈگۈنكى بۇ ھەيۋەتلىك ھەم مۇقەددەس مۇراسىمغا قاتناشقان پايتەختتىكى ھەر ساھە ۋەكىللىرى قاتتىق ھاياجانلاندى." into chinese''':
                                   '''参加今天这个庄严神圣的仪式，让首都各界代表个个感慨万千。'''},
                            {
                                '''Translate uyghur sentence "ئۇلار بۇ شىمغا تايىنىپ بۇ رايۇننىڭ نامىنى چىقىرىپ ، بەلگىلىك ئىقتىسادىي ئۈنۈم ياراتماقچى بولىۋاتىدۇ ." into chinese''':
                                    '''他们希望借助这条裤子提升该地区的知名度，并带来一定的经济效益。'''},
                            {
                                '''Translate uyghur sentence "جادكارى مۇنداق دېدى: جۇڭگونىڭ ئىقتىسادى ئىجتىمائىي تەرەققىياتىدا زور ئۆزگۈرۈشلەر بولدى، بىز بۇنىڭغا بەك قايىل." into chinese''':
                                    '''加德卡里表示，中国经济社会发生了巨大变化，我们对此表示钦佩。'''}

    ]


    # Step4: Build a prompt from the examples
    prompt.build_prompt(prompts, in_context_examples, n_shots=3)


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
    return lines


tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
input_texts = read_file('./ccmt/2021uc.ug.txt')
labels_list = read_file('./ccmt/2021uc.zh.txt')



all_bleu=0
f1 =open('./ccmt/pred/ug-zh/ug-zh-other-shot-gpt3.5-prompt3.txt', 'a', encoding='utf-8')
for labels,input_text in zip(labels_list,input_texts):
    # 调用翻译函数
    translations = translate('gpt-3.5-turbo', input_text)
    # 打印翻译结果
    f1.write(translations)
    f1.write('\n')
    print(f"Translation: {translations}")
    labels = tokenizer(labels,padding=True,truncation=True,return_tensors="pt")["input_ids"]
    translations = tokenizer(translations,padding=True,truncation=True,return_tensors="pt")["input_ids"]

    decoded_preds = tokenizer.batch_decode(translations, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds, labels = [], []
    metric = load_metric("sacrebleu")
    preds += [pred.strip() for pred in decoded_preds]
    labels += [[label.strip()] for label in decoded_labels]

    test_result = metric.compute(predictions=preds, references=labels)

    test_result = {"bleu": test_result['score']}
    print(test_result)
    all_bleu +=test_result['bleu']
bleu = all_bleu/len(labels_list)
print(bleu)