## About this app

**P52 · Self-Consistency Hallucination Detector.** If a model really knows an answer, asking it again gives the same
answer. If it is guessing, the answers change. This app measures that.

| Step | What happens |
|---|---|
| U1 | Your question is wrapped in a zero-shot, few-shot or chain-of-thought prompt at temperature T |
| U2 | The LLM (gpt-oss-20b on Groq) is sampled **5 times** |
| U3 | **RoBERTa-MNLI** checks every pair of answers for entailment in both directions. Image questions are answered by **BLIP-VQA** |
| U4 | Per-answer confidence chart + pairwise agreement heatmap |
| U5 | A warning appears when overall agreement is below the threshold, and the answer is shown as "Not sure" |

**Generative vs discriminative:** the LLM, BLIP and FLUX *generate* text or images. RoBERTa-MNLI is
*discriminative*: it only classifies a pair as entailment / neutral / contradiction.

**Limits:** consistency is not truth. A model can repeat the same wrong answer 5 times; that case is not caught.
Do not use this for medical, legal or financial decisions.

Licences: code MIT · gpt-oss Apache-2.0 · RoBERTa-MNLI MIT · BLIP BSD-3-Clause · FLUX.1-schnell Apache-2.0.
