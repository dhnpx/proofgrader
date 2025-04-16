import torch
from transformers import pipeline

# Copied the bolierplate from :
# https://github.com/deepseek-ai/DeepSeek-Math
# https://huggingface.co/deepseek-ai/deepseek-math-7b-rl

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "models/deepseek-math-7b-rl"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
model.generation_config = GenerationConfig.from_pretrained(model_name, local_file_only=True)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "The integral of x^2 from 0 to 2 is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
