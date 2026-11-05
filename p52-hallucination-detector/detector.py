"""
Self-Consistency Hallucination Detector - core logic (P52).

Idea: ask the SAME question several times with sampling turned on. If the model
really "knows" the answer, the samples agree. If it is guessing (hallucinating),
the samples drift apart. We measure that agreement with an NLI model.

    U1  prompts (zero-shot / few-shot / chain-of-thought) + temperature
    U2  multi-sampling: 5 answers per question from the LLM
    U3  agreement judged by RoBERTa-MNLI (bidirectional entailment);
        image questions are answered by BLIP-VQA instead of the text LLM
    U4  per-answer confidence chart + pairwise agreement heatmap
    U5  warning when agreement is low + input guardrail

Generative models used:  gpt-oss (Groq API), BLIP-VQA, FLUX.1-schnell (hosted image API)
Discriminative model:    RoBERTa-large-MNLI (classifies entail / neutral / contradict)
"""
import os
import re
import time
import math
import functools
import gc
import io
import urllib.parse
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
N_SAMPLES = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
BACKENDS = {
    "gpt-oss-20b (Groq, fast)": ("groq", "openai/gpt-oss-20b"),
    "gpt-oss-120b (Groq, stronger)": ("groq", "openai/gpt-oss-120b"),
}
DEFAULT_BACKEND = list(BACKENDS)[0]
TECHNIQUES = ["Zero-shot", "Few-shot", "Chain-of-thought"]

NLI_MODEL = os.getenv("NLI_MODEL", "FacebookAI/roberta-large-mnli")
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-vqa-base")
IMG_MODEL = os.getenv("IMG_MODEL", "black-forest-labs/FLUX.1-schnell")
QUANTIZE = os.getenv("QUANTIZE", "1") == "1"   # int8 weights -> fits the free 2.7 GB Streamlit host

# Colours (validated reference palette: status steps + one sequential blue)
C_GOOD, C_WARN, C_BAD = "#0ca30c", "#fab219", "#d03b3b"
C_INK, C_INK2, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HEAT_CMAP, HEAT_TEXT_HIGH, HEAT_TEXT_LOW = "Blues", "white", None  # the app overrides these for its dark theme


def _torch():
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch


# --------------------------------------------------------------------------
# U5 - Guardrail (runs BEFORE any model is called)
# --------------------------------------------------------------------------
_HARM = [
    r"\b(make|build|synthes\w*|cook|assemble)\b.{0,40}\b(bomb|explosive|nerve agent|bio ?weapon|meth|ricin|sarin)\b",
    r"\b(malware|ransomware|keylogger|ddos)\b",
    r"\b(kill|poison|hurt|stab|shoot)\b.{0,15}\b(a person|someone|people|my \w+|him|her|them)\b",
]
_SELF_HARM = r"\b(suicide|kill myself|self[- ]?harm|end my life)\b"
_INJECTION = r"\b(ignore|disregard)\b.{0,20}\b(previous|prior|above|all)\b.{0,20}\b(instructions|rules)\b"
MAX_LEN = 500


def guardrail(text):
    """Return (allowed: bool, message: str)."""
    t = (text or "").strip()
    if not t:
        return False, "Please type a question."
    if len(t) > MAX_LEN:
        return False, f"Question is too long (max {MAX_LEN} characters)."
    low = t.lower()
    if re.search(_SELF_HARM, low):
        return False, ("This tool only checks factual questions. If you are going through something "
                       "difficult, please talk to someone you trust or call Tele-MANAS (India): 14416.")
    if any(re.search(p, low) for p in _HARM):
        return False, "Blocked by safety guardrail: this request could cause harm."
    if re.search(_INJECTION, low):
        return False, "Blocked by safety guardrail: prompt-injection pattern detected."
    return True, ""


# --------------------------------------------------------------------------
# U1 - Prompt techniques
# --------------------------------------------------------------------------
SYSTEM = "You are a factual question-answering assistant."
FEW_SHOT = [
    ("What is the capital of France?", "Paris."),
    ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare."),
    ("What is the chemical symbol for sodium?", "Na."),
]


def build_messages(question, technique):
    if technique == "Few-shot":
        msgs = [{"role": "system", "content": SYSTEM + " Answer in the same short style as the examples."}]
        for q, a in FEW_SHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": question})
        return msgs
    if technique == "Chain-of-thought":
        return [{"role": "system", "content": SYSTEM + " Think step by step in at most three short sentences, "
                 "then write the final answer on the last line as 'Final answer: <answer>'."},
                {"role": "user", "content": question}]
    # Zero-shot (default)
    return [{"role": "system", "content": SYSTEM + " Reply with the answer only, in one short sentence."},
            {"role": "user", "content": question}]


def extract_answer(text, technique):
    """Pull the short final answer out of a raw generation."""
    text = (text or "").strip()
    if technique == "Chain-of-thought":
        found = re.findall(r"final answer\s*[:\-]\s*(.+)", text, flags=re.I)
        if found:
            text = found[-1]
        else:
            lines = [l for l in text.splitlines() if l.strip()]
            text = lines[-1] if lines else ""
    text = re.sub(r"\s+", " ", text.replace("**", "")).strip()
    return text[:300] if text else "(no answer)"


# --------------------------------------------------------------------------
# U2 - Multi-sampling
# --------------------------------------------------------------------------
def _groq_once(messages, temperature, model, max_tokens=800):
    key = os.getenv("GROQ_API_KEY")
    payload = {"model": model, "messages": messages, "temperature": float(temperature),
               "max_completion_tokens": max_tokens,
               "reasoning_effort": "low", "include_reasoning": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(8):
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=90)
        if r.status_code == 429:  # rate limit -> wait and retry
            wait = float(r.headers.get("retry-after", 2 * (attempt + 1)) or 2)
            time.sleep(min(max(wait, 1.0), 30.0))
            continue
        if r.status_code == 400 and "reasoning" in r.text and "reasoning_effort" in payload:
            payload.pop("reasoning_effort", None)
            payload.pop("include_reasoning", None)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        return (msg.get("content") or "").strip()
    raise RuntimeError("Groq rate limit reached - wait a minute and try again.")


def _quantize(model):
    """Dynamic int8 quantization of all Linear layers (about 3x less RAM, faster on CPU)."""
    if not QUANTIZE:
        return model
    torch = _torch()
    try:
        qd = torch.ao.quantization.quantize_dynamic
    except AttributeError:
        qd = torch.quantization.quantize_dynamic
    try:
        model = qd(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:  # if quantization is unavailable, keep float32
        print("Quantization skipped:", e)
    gc.collect()
    return model


def sample_llm(question, technique, temperature, backend, n=N_SAMPLES):
    """Return (raw_texts, backend_label_actually_used)."""
    _, model = BACKENDS.get(backend, BACKENDS[DEFAULT_BACKEND])
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it in the app's Secrets settings.")
    messages = build_messages(question, technique)
    with ThreadPoolExecutor(max_workers=n) as pool:
        raws = list(pool.map(lambda _: _groq_once(messages, temperature, model), range(n)))
    return raws, backend


# --------------------------------------------------------------------------
# U3 (multimodal) - BLIP-VQA for image questions
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _blip():
    from transformers import BlipProcessor, BlipForQuestionAnswering
    model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL, low_cpu_mem_usage=True).eval()
    return BlipProcessor.from_pretrained(BLIP_MODEL), _quantize(model)


def sample_blip(image, question, temperature, n=N_SAMPLES):
    torch = _torch()
    proc, model = _blip()
    inputs = proc(images=image.convert("RGB"), text=question, return_tensors="pt")
    answers = []
    for i in range(n):
        torch.manual_seed(1234 + i)  # different seed per sample, reproducible across runs
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=True, temperature=max(float(temperature), 0.05),
                                 top_k=50, max_new_tokens=20)
        answers.append(proc.decode(ids[0], skip_special_tokens=True).strip() or "(no answer)")
    return answers


# --------------------------------------------------------------------------
# U3 - Agreement with RoBERTa-MNLI
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _nli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    try:  # "eager" attention so we can read attention weights for the explanation tab
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, attn_implementation="eager",
                                                                   low_cpu_mem_usage=True)
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, low_cpu_mem_usage=True)
    model.eval()
    model = _quantize(model)
    labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    return tok, model, labels


def warmup():
    """Load the NLI model in the background so the first click is fast."""
    try:
        _nli()
    except Exception as e:  # never crash the app on warm-up
        print("NLI warm-up failed:", e)


def nli_probs(premises, hypotheses):
    """Return array (k, 3) of [P(entail), P(neutral), P(contradict)]."""
    torch = _torch()
    tok, model, lab = _nli()
    enc = tok(premises, hypotheses, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        p = model(**enc).logits.softmax(-1).numpy()
    return np.stack([p[:, lab["entailment"]], p[:, lab["neutral"]], p[:, lab["contradiction"]]], axis=1)


def normalize(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"^\s*(the\s+)?(final\s+)?answer(\s+is)?\s*[:\-]?\s*", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def agreement_matrix(question, answers):
    """
    E[i, j] = P(answer_i entails answer_j)   (both prefixed with the question)
    A[i, j] = (E[i, j] + E[j, i]) / 2         symmetric agreement in [0, 1]
    C[i, j] = P(answer_i contradicts answer_j)
    Identical answers (after normalisation) skip the NLI call and get 1.0.
    """
    n = len(answers)
    E, C = np.zeros((n, n)), np.zeros((n, n))
    norm = [normalize(a) for a in answers]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if norm[i] and norm[i] == norm[j]:
                E[i, j] = 1.0
            else:
                pairs.append((i, j))
    if pairs:
        ctx = lambda a: f"Question: {question} Answer: {a}"
        probs = nli_probs([ctx(answers[i]) for i, _ in pairs], [ctx(answers[j]) for _, j in pairs])
        for (i, j), p in zip(pairs, probs):
            E[i, j], C[i, j] = p[0], p[2]
    A = (E + E.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, E, C


def cluster(A, thr=0.5):
    """Group answers that mean the same thing (union-find on A >= thr)."""
    n = len(A)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] >= thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def score(A):
    n = len(A)
    conf = (A.sum(1) - 1.0) / (n - 1)                 # per-answer confidence
    agreement = float((A.sum() - n) / (n * (n - 1)))  # overall mean pairwise agreement
    groups = cluster(A)
    p = np.array([len(g) for g in groups], dtype=float) / n
    entropy = float(-(p * np.log(p)).sum() / math.log(n))  # semantic entropy, normalised 0..1
    best_group = groups[0]
    rep = max(best_group, key=lambda i: conf[i])
    return {"confidence": conf, "agreement": agreement, "clusters": groups,
            "majority_frac": len(best_group) / n, "semantic_entropy": entropy, "rep_index": int(rep)}


def level(agreement, threshold):
    if agreement < threshold:
        return "low"
    if agreement < max(threshold, 0.8):
        return "medium"
    return "high"


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def detect(question, technique="Zero-shot", temperature=0.9, backend=DEFAULT_BACKEND,
           image=None, threshold=0.5, n=N_SAMPLES):
    ok, msg = guardrail(question)
    if not ok:
        return {"blocked": True, "message": msg}
    t0 = time.time()
    if image is not None:
        raws = sample_blip(image, question, temperature, n)
        answers, used = raws, "BLIP-VQA (image question)"
    else:
        raws, used = sample_llm(question, technique, temperature, backend, n)
        answers = [extract_answer(r, technique) for r in raws]
    A, E, C = agreement_matrix(question, answers)
    s = score(A)
    return {"blocked": False, "question": question, "technique": technique, "temperature": temperature,
            "backend": used, "raws": raws, "answers": answers, "A": A, "E": E, "C": C,
            "threshold": threshold, "level": level(s["agreement"], threshold),
            "answer": answers[s["rep_index"]], "latency": time.time() - t0, **s}


# --------------------------------------------------------------------------
# U4 - Charts
# --------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(C_SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK2, labelsize=9)


def confidence_chart(res):
    conf, A, thr = res["confidence"], res["A"], res["threshold"]
    n = len(conf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(C_SURF)

    # Left: per-answer confidence bars
    _style(ax1)
    colors = [C_BAD if c < thr else (C_WARN if c < max(thr, 0.8) else C_GOOD) for c in conf]
    labels = [f"S{i+1}" + ("  ★" if i == res["rep_index"] else "") for i in range(n)]
    ax1.bar(range(n), conf, color=colors, width=0.55, zorder=3)
    ax1.axhline(thr, color=C_INK2, lw=1.2, ls="--", zorder=4)
    ax1.text(n - 0.5, 1.09, f"- - warning threshold {thr:.2f}", ha="right", va="top", fontsize=8, color=C_INK2)
    for i, c in enumerate(conf):
        ax1.text(i, c + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=9, color=C_INK)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.12)
    ax1.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax1.set_ylabel("confidence (mean NLI agreement)", color=C_INK2, fontsize=9)
    ax1.set_title(f"Per-answer confidence · overall {res['agreement']:.2f} ({res['level']})",
                  loc="left", fontsize=11, color=C_INK)

    # Right: pairwise agreement heatmap (sequential single hue)
    ax2.imshow(A, cmap=HEAT_CMAP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color=HEAT_TEXT_HIGH if A[i, j] > 0.6 else (HEAT_TEXT_LOW or C_INK))
    ticks = [f"S{i+1}" for i in range(n)]
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(ticks)
    ax2.set_yticklabels(ticks)
    ax2.tick_params(colors=C_INK2, labelsize=9, length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title("Pairwise agreement (bidirectional entailment)", loc="left", fontsize=11, color=C_INK)
    fig.tight_layout()
    return fig


def attention_figure(answer_a, answer_b):
    """Unit II: show which tokens of answer B attend to which tokens of answer A
    inside RoBERTa-MNLI (last layer, averaged over heads)."""
    torch = _torch()
    tok, model, lab = _nli()
    enc = tok(answer_a, answer_b, return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    probs = out.logits.softmax(-1)[0].numpy()
    att = out.attentions[-1][0].mean(0).numpy()
    ids = enc["input_ids"][0].tolist()
    toks = [t.replace("Ġ", "") or "␣" for t in tok.convert_ids_to_tokens(ids)]
    seps = [k for k, t in enumerate(ids) if t == tok.sep_token_id]
    if len(seps) >= 3:
        a_idx, b_idx = list(range(1, seps[0])), list(range(seps[1] + 1, seps[2]))
    else:
        a_idx = b_idx = list(range(1, len(ids) - 1))
    sub = att[np.ix_(b_idx, a_idx)]
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(a_idx) + 2), max(3, 0.35 * len(b_idx) + 1.6)))
    fig.patch.set_facecolor(C_SURF)
    ax.imshow(sub, cmap=HEAT_CMAP, aspect="auto")
    ax.set_xticks(range(len(a_idx)))
    ax.set_xticklabels([toks[k] for k in a_idx], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(b_idx)))
    ax.set_yticklabels([toks[k] for k in b_idx], fontsize=8)
    ax.tick_params(colors=C_INK2, length=0)
    ax.set_xlabel("Answer A tokens (premise)", color=C_INK2, fontsize=9)
    ax.set_ylabel("Answer B tokens (hypothesis)", color=C_INK2, fontsize=9)
    ax.set_title(f"RoBERTa-MNLI last-layer attention · entail {probs[lab['entailment']]:.2f} · "
                 f"neutral {probs[lab['neutral']]:.2f} · contradict {probs[lab['contradiction']]:.2f}",
                 loc="left", fontsize=9.5, color=C_INK)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Unit IV - Image generation (hosted FLUX.1-schnell; too heavy for the free CPU host)
# --------------------------------------------------------------------------
def illustrate(text, seed=0):
    """Try Hugging Face Inference Providers (if HF_TOKEN is set), else the free Pollinations API."""
    ok, msg = guardrail(text)
    if not ok:
        raise ValueError(msg)
    from PIL import Image
    prompt = f"clean flat educational illustration, {text}"
    errors = []
    if os.getenv("HF_TOKEN"):
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
            return client.text_to_image(prompt, model=IMG_MODEL, seed=int(seed)), "FLUX.1-schnell via Hugging Face"
        except Exception as e:
            errors.append(f"HF: {e}")
    try:
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt) +
               f"?width=512&height=512&seed={int(seed)}&nologo=true&model=flux")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB"), "FLUX via Pollinations"
    except Exception as e:
        errors.append(f"Pollinations: {e}")
    raise RuntimeError("Image generation failed - " + "; ".join(errors))
