from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer
import random
import sys
from transformers import TrainingArguments

def formatting_func(example):
    # prompt chosen rejected
    out = {"prompt": [], "chosen": [], "rejected": []}
    for type, correct, incorrect in zip(
        example["type"], example["correct"], example["incorrect"]
    ):
        nochange = random.choice([True, False])
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

training_args = TrainingArguments(
    per_device_train_batch_size=30,
    num_train_epochs=3,
    logging_steps=1,
    output_dir="dpo_trainer",
    save_steps=500
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