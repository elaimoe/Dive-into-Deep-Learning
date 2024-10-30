from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# 加载模型和分词器
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    # 将文本进行分词
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 进行翻译
    translated = model.generate(**inputs)
    # 解码翻译后的文本
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def translate_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    translated_lines = []
    for line in tqdm(lines, desc="Translating"):
        translated_line = translate_text(line.strip())
        translated_lines.append(translated_line + "\n")

    # 写入翻译结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(translated_lines)
    print("Translation completed. Output saved to", output_file)

# 使用方法：将 "input.txt" 替换为你的英文文件路径，"output.txt" 为翻译后的输出文件路径
translate_file("input.txt", "output.txt")

