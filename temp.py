from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/home/nfs05/shenyz/translation/verl/bs@2_@20251118_211452/global_step_140/huggingface"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "给定源文：'The MagDart Beauty Light is another magnetic attachment that comes with a light 'ring' that has 60 LEDs and flips up to illuminate your face while taking selfies.'，和它的翻译草稿：'MagDart Beauty Light 是一款带有“灯环”以及 60 个 LED 灯的磁性附件，可以将灯翻转，以拍摄你的脸为照明。'必须先理解源文，然后参考以下标准对草稿进行进一步修改润色\n\n1. 草稿翻译可能漏翻，请不要遗漏原文的含义\n\n2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序\n\n3. 请注意仔细理解语境，选择书面语还是口语化表达\n\n4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会\n\n5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会\n\n6. 注意你的润色对象是翻译后的草稿，不是源文\n\n7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考\n\n8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文\n\n9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻\n\n10. 可以有思考过程，但是最终回复中仅输出你润色后的内容\n\n请返回你最后的润色翻译文本，不要输出多余内容。"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

