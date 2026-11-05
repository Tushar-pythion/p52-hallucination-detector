# Model Card - Self-Consistency Hallucination Detector

## Overview
A system (not a newly trained model) that estimates whether an LLM answer is a hallucination by
sampling the same question 5 times and measuring semantic agreement between the samples with a
natural-language-inference (NLI) classifier.

| Component | Model | Type |
|---|---|---|
| Answer generator | openai/gpt-oss-20b (Groq API) | generative, decoder-only transformer |
| Image-question answerer | Salesforce/blip-vqa-base | generative, vision-language |
| Agreement judge | FacebookAI/roberta-large-mnli (int8 dynamic quantization) | discriminative, encoder-only transformer |
| Illustration | FLUX.1-schnell (hosted API) | generative, latent diffusion/flow transformer |

No model was fine-tuned. All adaptation is prompt-based (zero-shot, few-shot, chain-of-thought).

## Intended use
- Teaching and demonstrating hallucination detection.
- A "second opinion" signal for short factual questions.

## Not intended for
- Medical, legal, financial or safety-critical decisions.
- Proving that an answer is true. High agreement means the model is *consistent*, not *correct*.

## Method
1. Generate 5 answers at temperature T (default 0.9).
2. For each ordered pair (i, j), RoBERTa-MNLI gives P(entailment) of "Question + answer i" ⇒ "Question + answer j".
3. Agreement A[i,j] = mean of both directions. Per-answer confidence = mean agreement with the other 4.
4. Overall agreement = mean of all off-diagonal A. Answers with A ≥ 0.5 are grouped into meaning groups;
   semantic entropy is computed over group sizes.
5. If overall agreement < threshold (default 0.5) the app shows a warning and replies "Not sure".

## Evaluation
20 factual questions with known answers (8 easy, 12 long-tail). For each, *flagged* (agreement < threshold)
is compared with *actually wrong* (most-agreed answer does not contain the gold answer).
Results: see the PDF report and `results/benchmark_results.csv`.

## Limitations and failure modes
- **Consistent hallucinations are missed.** If the model is confidently wrong in the same way 5 times, agreement is high.
- **False alarms on correct but differently-worded answers.** NLI can score "1908" vs "It happened in 1908 in Siberia" as neutral.
- Answer correctness in the benchmark is checked by string matching against aliases, which can misjudge unusual phrasings.
- RoBERTa-MNLI is English-only; non-English questions are unreliable.
- Cost/latency: 5 LLM calls + 20 NLI pairs per question. int8 quantization of the NLI model can shift probabilities slightly.
- BLIP-VQA answers are very short and can be biased towards common objects and colours.

## Ethical considerations
- **Safety guardrail:** harmful requests (weapons, malware), self-harm mentions (shows a helpline instead) and
  prompt-injection patterns are blocked before any model is called.
- The app never shows a low-agreement answer as fact; it labels it "Not sure".
- Low-agreement answers are not illustrated, so the image generator never decorates a hallucination.
- No user data is stored. Questions are sent to the Groq API to be answered; illustration prompts go to the image API.
- Bias: the underlying LLMs and NLI model reflect biases of their training data (mostly English, Western sources),
  so long-tail questions about other regions are more likely to be hallucinated *and* flagged.

## Licences
Code: MIT. Models: see README (Apache-2.0, MIT, BSD-3-Clause).
