---
license: apache-2.0
base_model: unsloth/Qwen2.5-3B-bnb-4bit
library_name: peft
tags:
  - dpo
  - trl
  - peft
  - lora
  - alignment
  - vietnamese
datasets:
  - argilla/ultrafeedback-binarized-preferences-cleaned
  - 5CD-AI/Vietnamese-alpaca-cleaned
language:
  - vi
  - en
pipeline_tag: text-generation
---

# lab22-dpo-vn — DPO-aligned Qwen2.5-3B (VN)

> ⚠️ **Before upload:** copy this file to `adapters/dpo/README.md`, then fill every
> `__FILL__` with your own numbers from `adapters/dpo/dpo_metrics.json` and NB4.
> Delete this blockquote. Upload with:
> `huggingface-cli upload <username>/lab22-dpo-vn ./adapters/dpo`

LoRA adapter trained with **Direct Preference Optimization (DPO)** on top of an
SFT-mini checkpoint, for VinUni AICB Track-3 Day-22 Alignment Lab. Goes from SFT →
preference learning to improve helpfulness/safety on Vietnamese + English prompts.

## How it was built

1. **SFT-mini** (`adapters/sft-mini`) — LoRA r=16, α=32 on 1k `5CD-AI/Vietnamese-alpaca-cleaned`, 1 epoch.
2. **DPO** (this adapter) — TRL `DPOTrainer` on the SFT model + frozen reference, on 2k
   `argilla/ultrafeedback-binarized-preferences-cleaned` pairs.

## Hyperparameters

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit NF4) |
| Method | DPO (LoRA, r=16, α=32) |
| β | 0.1 |
| Learning rate | 5e-7 |
| Epochs | 1 |
| Preference pairs | 2000 (UltraFeedback) |
| max_length | 512 |
| Compute | __FILL__ (e.g. free Colab T4 16GB) |

## Results

| Metric | Value |
|---|---|
| End reward gap (chosen − rejected) | __FILL__ |
| Final chosen reward | __FILL__ |
| Final rejected reward | __FILL__ |
| NB4 win/loss/tie (8 prompts, judge=__FILL__) | __FILL__ |

> Note on reward curves: __FILL one sentence — did chosen rise, or did the gap grow
> mainly because rejected fell (likelihood displacement, deck §3.4)?__

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "unsloth/Qwen2.5-3B-bnb-4bit"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, "<username>/lab22-dpo-vn")

msgs = [{"role": "user", "content": "Giải thích ngắn gọn cách hoạt động của DPO."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
print(tok.decode(model.generate(ids, max_new_tokens=256)[0][ids.shape[1]:], skip_special_tokens=True))
```

GGUF quantizations (Q4_K_M + Q5_K_M) for llama.cpp are in the `gguf/` folder of this repo.

## Limitations

3B-scale adapter trained on a small slice for an educational lab — not production-grade.
Expect alignment-tax trade-offs (chat formatting up, some reasoning benchmarks flat/down).

## Citation / acknowledgments

Trained with Unsloth + TRL + PEFT for VinUniversity AICB Track-3 Day-22.
Datasets: UltraFeedback (Argilla), Vietnamese-alpaca-cleaned (5CD-AI).
