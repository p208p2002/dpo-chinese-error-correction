# DPO Chinese Error Correction
使用 Direct Preference Optimization (DPO) 訓練中文糾錯模型。

### Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,AddedToken
import sys

mode_id = "p208p2002/bloom-1b1-zh-error-correction-dpo"
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(mode_id)
tokenizer = AutoTokenizer.from_pretrained(mode_id)

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

    print("input :",input_text)
    print("output:",output_text)
    print('-'*30)
```
<!-- ```
input: <s>為了潔約能源請隨守關閉沒有使用的電器 </s>
output: <s>為了節約能源請隨時關閉沒有使用的電器 </s>
------------------------------
input: <s>今天新情很好 </s>
output: <s>今天心情很好 </s>
------------------------------
input: <s>你快樂我也很高心 </s>
output: <s>你快樂我也很高興 </s>
------------------------------
input: <s>但不再算再找實習生了 </s>
output: <s>但不再去找實習生了 </s>
------------------------------
input: <s>今天太陽很大要注意篩傷 </s>
output: <s>今天太陽很大要注意一下 </s>
------------------------------
input: <s>你要不要和我依起去台北 </s>
output: <s>你要不要和我一起去台北 </s>
------------------------------
input: <s>清晨六點終太陽會升起 </s>
output: <s>清晨六點鐘太陽會升起 </s>
------------------------------
input: <s>傾城六點鐘太陽會升起 </s>
output: <s>凌晨六點鐘太陽會升起 </s>
------------------------------
input: <s>鍋馬路時你應該要注意虹綠燈 </s>
output: <s>過馬路時你應該要注意紅綠燈 </s>
------------------------------
input: <s>他正在學學彈吉他 </s>
output: <s>他正在學習彈吉他 </s>
------------------------------
input: <s>下樓梯請注意階梯 </s>
output: <s>下樓梯請注意階梯 </s>
------------------------------
input: <s>此信件為系統自動發送之通知 </s>
output: <s>此信件為系統自動發送之通知 </s>
------------------------------
input: <s>此信件為系統自動發送知通知 </s>
output: <s>此信件為系統自動發送通知 </s>
------------------------------
input: <s>如為誤傳也請立即刪除本郵件並通知寄件者 </s>
output: <s>如為誤傳也請立即刪除本郵件並通知寄件者 </s>
------------------------------
(venv) philip@nca100-3-G1:~/ec-dpo$ python test_model.py dpo_trainer/checkpoint-250
input : <s>為了潔約能源請隨守關閉沒有使用的電器 </s>
output: <s>為了節約能源請隨時關閉沒有使用的電器 </s>
------------------------------
input : <s>今天新情很好 </s>
output: <s>今天心情很好 </s>
------------------------------
input : <s>你快樂我也很高心 </s>
output: <s>你快樂我也很高興 </s>
------------------------------
input : <s>但不再算再找實習生了 </s>
output: <s>但不再去找實習生了 </s>
------------------------------
input : <s>今天太陽很大要注意篩傷 </s>
output: <s>今天太陽很大要注意一下 </s>
------------------------------
input : <s>你要不要和我依起去台北 </s>
output: <s>你要不要和我一起去台北 </s>
------------------------------
input : <s>清晨六點終太陽會升起 </s>
output: <s>清晨六點鐘太陽會升起 </s>
------------------------------
input : <s>傾城六點鐘太陽會升起 </s>
output: <s>凌晨六點鐘太陽會升起 </s>
------------------------------
input : <s>鍋馬路時你應該要注意虹綠燈 </s>
output: <s>過馬路時你應該要注意紅綠燈 </s>
------------------------------
input : <s>他正在學學彈吉他 </s>
output: <s>他正在學習彈吉他 </s>
------------------------------
input : <s>下樓梯請注意階梯 </s>
output: <s>下樓梯請注意階梯 </s>
------------------------------
input : <s>此信件為系統自動發送之通知 </s>
output: <s>此信件為系統自動發送之通知 </s>
------------------------------
input : <s>此信件為系統自動發送知通知 </s>
output: <s>此信件為系統自動發送通知 </s>
------------------------------
input : <s>如為誤傳也請立即刪除本郵件並通知寄件者 </s>
output: <s>如為誤傳也請立即刪除本郵件並通知寄件者 </s>
------------------------------
``` -->
