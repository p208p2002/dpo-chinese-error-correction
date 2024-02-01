from transformers import AutoModelForCausalLM,TrainingArguments,AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("p208p2002/zhtw-sentence-error-correction", "alpha")
model = AutoModelForCausalLM.from_pretrained("p208p2002/llama-traditional-chinese-120M")
tokenizer = AutoTokenizer.from_pretrained("p208p2002/llama-traditional-chinese-120M")

def formatting_func(example):
    out = []
    for correct,incorrect in zip(example["correct"],example["incorrect"]):
        out.append(tokenizer.eos_token + incorrect + tokenizer.bos_token + correct + tokenizer.bos_token)
    return out

training_args = TrainingArguments(
    per_device_train_batch_size=50,
    num_train_epochs=3,
    logging_steps=100,
    output_dir="sft_trainer",
    save_steps=2000
)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset["train"],
    max_seq_length=256,
    formatting_func=formatting_func,
    dataset_kwargs={"add_special_tokens":False}
)

trainer.train()