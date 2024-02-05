from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,AddedToken
import sys

model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(sys.argv[-1])
tokenizer = AutoTokenizer.from_pretrained(sys.argv[-1])

test_texts = [
    "為了潔約能源請隨守關閉沒有使用的電器",
    "今天新情很好",
    "你快樂我也很高心",
    "但不再算再找實習生了",
    "今天太陽很大要注意篩傷",
    "你要不要和我依起去台北",
    "清晨六點終太陽會升起",
    "傾城六點鐘太陽會升起",
    "鍋馬路時你應該要注意虹綠燈",
    "他正在學學彈吉他",
    "下樓梯請注意階梯",
    "此信件為系統自動發送之通知",
    "此信件為系統自動發送知通知",
    "如為誤傳也請立即刪除本郵件並通知寄件者"
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
    )
    decode_out = tokenizer.decode(out[0])

    input_text,output_text = decode_out.split("\n") 
    input_text = input_text.strip()
    output_text = output_text.strip()

    print(input_text)
    print(output_text)
    print('-'*30)

