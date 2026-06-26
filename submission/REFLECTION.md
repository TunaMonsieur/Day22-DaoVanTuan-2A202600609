# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Họ Tên>_
**Cohort:** _<A20-K1 / A20-K2 / ...>_
**Tier đã chạy:** _<T4 | BIGGPU | both>_
**Date:** _<YYYY-MM-DD>_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | _<e.g., Free Colab T4 16GB / RTX 4060 8GB / A100 40GB>_ |
| CUDA / driver | _<e.g., CUDA 12.1, driver 535>_ |
| Base model | _<e.g., unsloth/Qwen2.5-3B-bnb-4bit>_ |
| SFT dataset slice | _<e.g., 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch>_ |
| Preference dataset slice | _<e.g., argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch>_ |
| `COMPUTE_TIER` env | _<T4 | BIGGPU>_ |
| Total cost | _<e.g., $0 (free Colab) / $1.20 (Colab Pro A100 30 min)>_ |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _<e.g., 28 min>_ |
| VRAM peak | _<e.g., 10.4 GB>_ | _<e.g., 13.8 GB>_ |
| Final loss | _<e.g., 1.82 (SFT)>_ | _<e.g., 0.48 (DPO)>_ |
| Reward gap (chosen − rejected, end of training) | n/a | _<e.g., 1.34>_ |
| Mean output length | _<e.g., 142 tokens>_ | _<e.g., 87 tokens (-39%)>_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste `03_dpo_reward_curves.png` here** (or link to it in `submission/screenshots/`).

_Interpret both `chosen_rewards` and `rejected_rewards` separately. Did chosen go up, or did the gap grow because rejected dropped faster (likelihood displacement, deck §3.4)? What does this tell you about whether DPO did what you wanted? Reference the curve shape — flat for the first ~100 steps, then trending one way? KL divergence to reference at end?_

_Answer here. ≥ 100 words._

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | _<...>_ | _<...>_ | _<...>_ | _<SFT \| DPO \| tie>_ |
| 2 | helpfulness | | | | |
| 3 | helpfulness | | | | |
| 4 | helpfulness | | | | |
| 5 | safety | | | | |
| 6 | safety | | | | |
| 7 | safety | | | | |
| 8 | safety | | | | |

**Win/loss/tie summary:** _<e.g., SFT+DPO wins 5/8, ties 2/8, loses 1/8>_

**Judge used:** _<gpt-4o-mini | claude-haiku-4-5 | manual rubric>_

---

## 5. β trade-off  (β-sweep bonus, rigor add-on +6)

> **Paste `bonus-beta-sweep.png` here** (reward gap & win-rate vs β).

Ran NB3 three times via `make beta-sweep` with β ∈ {0.05, 0.1, 0.5}, all else fixed
(lr=5e-7, 1 epoch, same 2k UltraFeedback slice). Numbers from each run's
`dpo_metrics.json` + NB4 judge on the 8 fixed prompts:

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | __FILL__ | __FILL__ | __FILL__ | most aggressive — least KL constraint |
| 0.1 (default) | __FILL__ | __FILL__ | __FILL__ | deck §5.2 default |
| 0.5 | __FILL__ | __FILL__ | __FILL__ | most conservative — strongest tether to ref |

**Interpretation (fill ≥100 words, edit the bracketed claims to match my numbers):**

The sweet spot for my data was **β = __FILL__**. At low β (0.05) the policy is least
constrained by the reference, so I expected — and observed __[did / did not]__ — the
largest reward gap but also __[more / less]__ length drift and __[higher / lower]__
win-rate, since a loose KS tether lets the model exploit the preference signal
(including length hacking, deck §3.4). At high β (0.5) the model stays close to the SFT
reference, giving a __[smaller]__ reward gap but __[more stable / more conservative]__
generations. This __[matches / partly matches / contradicts]__ the deck §3.3 prediction
that β trades off how aggressively the policy moves from the reference: small β = faster
preference fit but higher displacement risk, large β = safer but weaker alignment. The
reason β = __FILL__ won for me is __[my data / my judge / my prompt mix]__: __FILL one
sentence on why__.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

_Answer here. ≥ 150 words._

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _<...>_ | _<...>_ | _<...>_ |
| GSM8K | _<...>_ | _<...>_ | _<...>_ |
| MMLU (sampled) | _<...>_ | _<...>_ | _<...>_ |
| AlpacaEval-lite | _<...>_ | _<...>_ | _<...>_ |

_Interpret the deltas. Which benchmark went up most? Did GSM8K or MATH regress (alignment tax — see deck §8.1)? Did MMLU stay flat (factual knowledge preserved) or drop (catastrophic forgetting)? Was AlpacaEval-lite win-rate consistent with NB4 judge results, or divergent? Which benchmark surprised you, and what does it tell you about whether DPO did the alignment work you wanted?_

_Answer here. ≥ 150 words._

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu)_
