# 🔍 Self-Consistency Hallucination Detector (P52)

**Goal:** detect likely hallucinations by asking an LLM the same question several times.
If the model knows the answer, its samples agree. If it is guessing, they drift apart.

- **Live app:** `<add Streamlit app URL>`
- **Demo video:** `<add link>`

## How it works

```
question ──► guardrail ──► prompt (zero-shot / few-shot / CoT, temperature T)
                                   │
                    ┌──────────────┴──────────────┐
             text question                  image question
        LLM × 5 samples                     BLIP-VQA × 5 samples
        (Groq gpt-oss-20b)
                    └──────────────┬──────────────┘
                                   ▼
          RoBERTa-MNLI: P(entail) for every ordered pair (20 pairs)
          agreement A[i,j] = (P(i⇒j) + P(j⇒i)) / 2
                                   ▼
   per-answer confidence · overall agreement · meaning groups · semantic entropy
                                   ▼
            confidence chart  +  warning banner if agreement < threshold
```

| Unit task | Where in the code |
|---|---|
| **U1** Sample 5 answers per question at a set temperature (3 prompt techniques) | `detector.build_messages`, temperature slider |
| **U2** Multi-sampling with the LLM | `detector.sample_llm` (5 parallel Groq API calls) |
| **U3** Judge agreement with RoBERTa-MNLI; BLIP for image questions | `detector.agreement_matrix`, `detector.sample_blip` |
| **U4** Confidence chart per answer | `detector.confidence_chart` |
| **U5** Warn when agreement is low | warning banners in `app.py` + `detector.guardrail` |
| Deliverable: flagged vs actually-wrong on 20 questions | `benchmark.py` and the **Benchmark** tab |

### Five common layers
| Layer | What this project does |
|---|---|
| I Foundations | Lifecycle diagram (`docs/lifecycle.png`); generative (LLM, BLIP, diffusion) vs discriminative (RoBERTa-MNLI) framing; 3 prompt techniques compared; probability view (NLI probabilities, semantic entropy) |
| II GANs & LLMs | Transformer LLM at the core; prompt-based (in-context) learning compared: zero-shot vs few-shot vs CoT; **attention explanation** tab (RoBERTa-MNLI attention heatmap) |
| III GPT, BERT, multimodal | GPT-style decoder (gpt-oss), BERT-style encoder (RoBERTa), multimodal BLIP-VQA for image questions |
| IV Tools & generation | Python + PyTorch (NLI and BLIP run locally, int8-quantized); **image generation** with FLUX.1-schnell through a hosted API in the *Illustrate* tab |
| V Responsible AI | Measured hallucination test (20-question benchmark), safety guardrail, licence/IP notes, `MODEL_CARD.md` |

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run app.py
```
Benchmark from the command line:
```bash
python benchmark.py --techniques Zero-shot Few-shot Chain-of-thought
```

## Models and licences
| Model | Role | Licence |
|---|---|---|
| openai/gpt-oss-20b / 120b via Groq API | text generation | Apache-2.0 (weights); Groq API terms |
| FacebookAI/roberta-large-mnli | agreement judge (NLI) | MIT |
| Salesforce/blip-vqa-base | image question answering | BSD-3-Clause |
| black-forest-labs/FLUX.1-schnell (via Pollinations or HF Inference Providers) | illustration | Apache-2.0 |

Code is MIT-licensed. Generated images are for illustration only.
The API key is stored in Streamlit Cloud **Secrets**, never in the code.

## Deployment
Streamlit Community Cloud (free): connect this GitHub repo, main file `app.py`, add `GROQ_API_KEY` in Secrets.
RoBERTa-MNLI and BLIP are int8-quantized so both fit in the free host's memory.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, limitations and ethics.
