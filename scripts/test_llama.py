import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# 用于验证从 modelscope 下载的 llama 模型是否完整

# 指定模型路径
path = ""
model_path = "../autodl-pub/Meta-Llama-3.1-8B-Instruct"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path)

# 如果使用 GPU，将模型和输入数据移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入文本
input_text = "Once upon a time"

# 分词
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 生成文本
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码生成的文本
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)