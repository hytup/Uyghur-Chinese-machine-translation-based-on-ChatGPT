import openai
from datasets import load_metric
from transformers import AutoTokenizer


# 设置OpenAI API密钥
openai.api_key = "XXXXXXXXXXXXXXX"



# 定义翻译任务
def translate(model , input_text , target_lang="Chinese"):

    source_lang = "uyghur"

    prompt = f"Translate {source_lang} sentence {input_text} into {target_lang}: "

    if model == 'gpt-3.5-turbo':

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        translation = response.choices[0].message.content.strip()
    else:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0,
        )
        translation = response.choices[0].text.strip()
    return translation

def read_file(root):
    lines =[]
    file = open(root,'r',encoding='utf-8')
    for line in file.readlines():
        line = line.strip()
        lines.append(line)
    return lines


tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

input_texts = read_file('./test-data/uyghur.txt') # 源语言文件
labels_list = read_file('./test-data/Chinese.txt') # 目标语言文件


all_bleu=0
f1 =open('./data/pred/pred-ug-zh.txt','a',encoding='utf-8') # 预测文件
for labels,input_text in zip(labels_list,input_texts):
    # 调用翻译函数
    translations = translate('text-davinci-002', input_text)
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