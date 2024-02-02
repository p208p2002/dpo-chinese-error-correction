from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,AddedToken
import sys

model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(sys.argv[-1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[-1])

test_texts = [
    "你要不要和我依起去台北",
    "清晨六點終太陽會升起",
    "過馬路時你應該要注意虹綠燈",
    "他正在學學彈吉他"
]
for text in test_texts:
    inputs = tokenizer(
        f"{tokenizer.bos_token}{text} {tokenizer.eos_token}\n {tokenizer.bos_token}",
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]

    out = model.generate(
        inputs,
        max_new_tokens=20,
        # early_stopping=True,
        # do_sample=False
    )
    print(tokenizer.decode(out[0]))
    # print(tokenizer.eos_token_id)
    # print(out[0])
