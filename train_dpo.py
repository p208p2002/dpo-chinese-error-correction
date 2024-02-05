from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer
import sys
from transformers import TrainingArguments

def formatting_func(example):
    # prompt chosen rejected
    out = {"prompt": [], "chosen": [], "rejected": []}
    for i,(type, correct, incorrect) in enumerate(zip(
        example["type"], example["correct"], example["incorrect"]
    )):
        nochange = i % 2
        if nochange:
            # 正確的不要改成錯的
            out["prompt"].append(correct)
            out["chosen"].append(correct)
            out["rejected"].append(incorrect)

        else:
            # 有錯誤的資料需要修正
            out["prompt"].append(incorrect)
            out["chosen"].append(correct)
            out["rejected"].append(incorrect)

    return out

dataset = load_dataset("p208p2002/zhtw-sentence-error-correction", "gamma")
dataset_beta_rlhf = dataset.map(
    formatting_func, remove_columns=["type", "correct", "incorrect"], batched=True
)
model = AutoModelForCausalLM.from_pretrained(sys.argv[-1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[-1])
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    per_device_train_batch_size=5,
    gradient_accumulation_steps=10,
    num_train_epochs=3,
    logging_steps=1,
    output_dir="dpo_trainer",
    save_steps=50,
    remove_unused_columns=False,
    learning_rate=5e-7,
    max_steps=1000
)

dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset_beta_rlhf["train"],
    tokenizer=tokenizer,
    max_length=256,
)

dpo_trainer.train()