from transformers import AutoModelForCausalLM,TrainingArguments,AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AddedToken

dataset = load_dataset("p208p2002/zhtw-sentence-error-correction", "alpha")
MODEL_NAME_OR_ID = "ckip-joint/bloom-1b1-zh"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_ID)
tokenizer.padding_side = "right"
tokenizer.pad_token = "<pad>"

def formatting_func(example):
    out = []
    for correct,incorrect in zip(example["correct"],example["incorrect"]):
        out.append(f"{tokenizer.bos_token}{incorrect} {tokenizer.eos_token}\n {tokenizer.bos_token}{correct} {tokenizer.eos_token}")
    return out

training_args = TrainingArguments(
    per_device_train_batch_size=5,
    gradient_accumulation_steps=10,
    num_train_epochs=3,
    logging_steps=100,
    output_dir="sft_trainer",
    save_steps=500
)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    max_seq_length=256,
    formatting_func=formatting_func,
    dataset_kwargs={
        "add_special_tokens":False,
    }
)

trainer.train()