from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import sys

model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(sys.argv[-1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[-1])
inputs = tokenizer(
    f"{tokenizer.bos_token}一起去完遊戲吧{tokenizer.eos_token}",
    return_tensors="pt",
    add_special_tokens=False,
)["input_ids"]
out = model.generate(inputs)
print(tokenizer.decode(out[0]))
