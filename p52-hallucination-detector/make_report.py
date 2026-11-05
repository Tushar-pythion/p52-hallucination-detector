"""
Builds the CA report PDF from the benchmark CSV.

    python make_report.py --name Rahul --roll 52 \
        --app-url https://<app>.streamlit.app \
        --github https://github.com/<user>/<repo> \
        --csv results/benchmark_results.csv

Needs: pip install reportlab pandas matplotlib numpy requests
"""
import argparse
import datetime as dt
import os

import pandas as pd
from xml.sax.saxutils import escape as X
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
if os.path.exists(os.path.join(FONT_DIR, "DejaVuSans.ttf")):
    pdfmetrics.registerFont(TTFont("Body", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")
    BODY, BOLD = "Body", "Body-Bold"
else:
    BODY, BOLD = "Helvetica", "Helvetica-Bold"

BLUE = colors.HexColor("#1c5cab")
INK2 = colors.HexColor("#52514e")
GRID = colors.HexColor("#d9d8d4")
HEAD_BG = colors.HexColor("#eef4fc")
RED = colors.HexColor("#d03b3b")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", fontName=BOLD, fontSize=22, leading=28, alignment=TA_CENTER, textColor=BLUE),
    "sub": ParagraphStyle("s", fontName=BODY, fontSize=13, leading=18, alignment=TA_CENTER, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=15, leading=20, spaceBefore=14, spaceAfter=6, textColor=BLUE),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12, leading=16, spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName=BODY, fontSize=10, leading=14.5, spaceAfter=6),
    "small": ParagraphStyle("sm", fontName=BODY, fontSize=8.5, leading=11, textColor=INK2),
    "cell": ParagraphStyle("c", fontName=BODY, fontSize=8, leading=10),
    "cellb": ParagraphStyle("cb", fontName=BOLD, fontSize=8, leading=10),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5, leading=11, backColor=colors.HexColor("#f4f4f2"),
                           borderPadding=6, spaceAfter=8, spaceBefore=4),
    "warn": ParagraphStyle("w", fontName=BOLD, fontSize=10, leading=14, textColor=RED, spaceAfter=6),
}


def P(text, style="p"):
    return Paragraph(text, S[style])


def table(rows, widths, header=True, font_size=8):
    data = [[c if hasattr(c, "wrap") else Paragraph(X(str(c)), S["cellb" if header and r == 0 else "cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    t.setStyle(TableStyle(style))
    return t


def img(path, width):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 8)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "P52 · Self-Consistency Hallucination Detector")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build(a):
    W = A4[0] - 4 * cm
    have = a.csv and os.path.exists(a.csv)
    df = pd.read_csv(a.csv) if have else None
    story = []

    # ---------------- Title page ----------------
    story += [Spacer(1, 3 * cm), P("Generative AI · Continuous Assessment Mini Project", "sub"), Spacer(1, 0.6 * cm),
              P("P52 · Self-Consistency Hallucination Detector", "title"), Spacer(1, 0.4 * cm),
              P("Detecting likely LLM hallucinations by asking the same question several times", "sub"),
              Spacer(1, 2 * cm)]
    story.append(table([["Name", a.name], ["Roll No.", a.roll], ["Problem", "P52 (Track E: Ethics, governance and "
                        "responsible deployment)"], ["Live app", a.app_url], ["GitHub", a.github],
                        ["Date", dt.date.today().strftime("%d %B %Y")]], [4 * cm, W - 4 * cm], header=False))
    story.append(PageBreak())

    # ---------------- 1. Problem ----------------
    story.append(P("1. Problem statement", "h1"))
    story.append(P("<b>Goal:</b> detect likely hallucinations by asking the same question several times."))
    story.append(table([["Task", "Requirement"],
                        ["U1", "Sample 5 answers per question at a set temperature."],
                        ["U2", "Run multi-sampling with the LLM."],
                        ["U3", "Judge agreement with RoBERTa-MNLI; use BLIP for image questions."],
                        ["U4", "Show a confidence chart per answer."],
                        ["U5", "Warn the user when agreement is low."],
                        ["Deliverables", "Detector app; flagged vs actually-wrong table on 20 questions."]],
                       [3 * cm, W - 3 * cm]))
    story.append(P("2. Background", "h1"))
    story.append(P("A <b>hallucination</b> is fluent, confident text that is not supported by facts. Large language "
                   "models produce them because they are trained to predict likely next tokens, not to check truth. "
                   "When a model has strong evidence for an answer, the probability mass is concentrated, so repeated "
                   "sampling returns the same meaning. When it is guessing, the mass is spread out and samples disagree. "
                   "This idea underlies SelfCheckGPT (Manakul et al., 2023) and semantic entropy (Kuhn et al., 2023; "
                   "Farquhar et al., 2024). It needs no access to the model's weights or training data, so it also "
                   "works on API-only models."))
    story.append(P("<b>Generative vs discriminative framing.</b> The answer generators (gpt-oss, BLIP-VQA) "
                   "and the illustrator (FLUX.1-schnell) are <i>generative</i>: they model p(output | input) and sample new "
                   "text or images. RoBERTa-MNLI is <i>discriminative</i>: it models p(label | premise, hypothesis) "
                   "over three labels (entailment, neutral, contradiction) and never produces new content. The "
                   "project uses a discriminative model to audit a generative one."))

    # ---------------- 3. Lifecycle ----------------
    story.append(P("3. Generative-AI lifecycle", "h1"))
    lc = os.path.join(HERE, "docs", "lifecycle.png")
    if os.path.exists(lc):
        story.append(img(lc, W * 0.85))
    story.append(P("Figure 1. Lifecycle of the project, from scoping to deployment and governance.", "small"))

    # ---------------- 4. Design ----------------
    story.append(PageBreak())
    story.append(P("4. System design", "h1"))
    story.append(Paragraph(
        "question -&gt; guardrail -&gt; prompt (zero-shot | few-shot | CoT, temperature T)<br/>"
        "&nbsp;&nbsp;text question&nbsp; -&gt; LLM x 5 samples (gpt-oss-20b via Groq API)<br/>"
        "&nbsp;&nbsp;image question -&gt; BLIP-VQA x 5 samples<br/>"
        "-&gt; RoBERTa-MNLI on all 20 ordered answer pairs -&gt; agreement matrix A<br/>"
        "-&gt; confidence per answer, overall agreement, meaning groups, semantic entropy<br/>"
        "-&gt; confidence chart + warning banner if agreement &lt; threshold", S["code"]))
    story.append(table([
        ["Component", "Model / tool", "Role"],
        ["Answer generator", "openai/gpt-oss-20b via Groq API (gpt-oss-120b selectable)",
         "Decoder-only transformer (GPT-style). Generates 5 answers."],
        ["Image answerer", "Salesforce/blip-vqa-base", "Vision-language model; answers questions about an uploaded image."],
        ["Agreement judge", "FacebookAI/roberta-large-mnli", "Encoder-only transformer (BERT-style), 355M params, NLI classifier; int8-quantized on CPU."],
        ["Illustrator", "black-forest-labs/FLUX.1-schnell via hosted API (Pollinations / HF Inference Providers)", "Few-step latent generator; illustrates high-agreement answers."],
        ["UI / hosting", "Streamlit on Streamlit Community Cloud (free)", "Public live app; API key in Streamlit secrets."],
    ], [3.2 * cm, 6.3 * cm, W - 9.5 * cm]))

    story.append(P("5. Implementation of U1-U5", "h1"))
    story.append(P("U1 - Prompt techniques and temperature", "h2"))
    story.append(table([
        ["Technique", "Prompt design"],
        ["Zero-shot", "System: factual assistant, 'reply with the answer only, in one short sentence'. User: question."],
        ["Few-shot", "Same system role + three solved examples (capital of France, author of Romeo and Juliet, "
                     "symbol for sodium) as user/assistant turns, then the question. Teaches the short answer format "
                     "by in-context learning."],
        ["Chain-of-thought", "Think step by step in at most three sentences, then write 'Final answer: ...'. "
                             "Only the final-answer line is compared."],
    ], [3.2 * cm, W - 3.2 * cm]))
    story.append(P("Temperature is set with a slider (default 0.9). Sampling is required: at T = 0 every sample "
                   "would be identical and agreement would always be 1. The same temperature is used for all 5 samples."))
    story.append(P("U2 - Multi-sampling", "h2"))
    story.append(P("The 5 requests are sent to the Groq API in parallel threads; HTTP 429 rate-limit responses are "
                   "retried with back-off. gpt-oss is called with low reasoning effort so each answer stays short. "
                   "Fine-tuning was not used: prompt-based (in-context) learning adapts the model with no training "
                   "cost, which suits a free CPU host, and the comparison of zero-shot, few-shot and chain-of-thought "
                   "prompting is reported in Section 7."))
    story.append(P("U3 - Agreement with RoBERTa-MNLI, BLIP for images", "h2"))
    story.append(P("Each answer is written as <i>'Question: q Answer: a'</i> so short answers such as '1908' keep their "
                   "meaning. For every ordered pair (i, j) RoBERTa-MNLI gives P<sub>ent</sub>(i -&gt; j). Then:"))
    story.append(Paragraph(
        "A[i,j] = ( P_ent(i-&gt;j) + P_ent(j-&gt;i) ) / 2&nbsp;&nbsp;&nbsp;&nbsp;(identical answers: 1)<br/>"
        "confidence_i = mean over j != i of A[i,j]<br/>"
        "agreement = mean of all off-diagonal A[i,j]<br/>"
        "meaning groups: answers linked when A[i,j] &gt;= 0.5 (union-find)<br/>"
        "semantic entropy = - sum_g p_g log p_g / log 5,&nbsp; p_g = |group g| / 5", S["code"]))
    story.append(P("These are probabilities, so the app gives a <b>probability view</b> of the model's uncertainty "
                   "rather than a yes/no. For image questions the uploaded image and question go to BLIP-VQA, which "
                   "is sampled 5 times (different seeds); the same NLI agreement is then applied."))
    story.append(P("U4 - Confidence chart", "h2"))
    story.append(P("For every question the app draws (a) a bar per sample showing its confidence, coloured "
                   "green / amber / red with a text level, plus a dashed threshold line, and (b) a 5 x 5 heatmap of "
                   "pairwise agreement. The most-agreed answer is marked with a star."))
    story.append(P("U5 - Low-agreement warning and guardrail", "h2"))
    story.append(P("If agreement &lt; threshold (default 0.5, adjustable) a red banner says <i>'Low agreement - likely "
                   "hallucination'</i> and the app replies 'Not sure' instead of presenting the answer as fact. "
                   "Between the threshold and 0.8 an amber 'check before using' banner is shown. A rule-based "
                   "guardrail runs before any model call and blocks harmful requests, prompt-injection phrases and "
                   "over-long inputs; a self-harm mention gets a helpline message instead of an answer."))

    story.append(P("6. Transformer internals and image generation", "h1"))
    story.append(P("<b>Attention explanation.</b> The 'Why? (attention)' tab feeds two sampled answers to RoBERTa-MNLI "
                   "as a premise/hypothesis pair and plots the last-layer self-attention (averaged over 16 heads) from "
                   "hypothesis tokens to premise tokens. Each head computes softmax(QK<super>T</super>/sqrt(d))V, so a "
                   "bright cell means the token in answer B drew information from that token in answer A. When two "
                   "answers name different people or years, attention typically concentrates on those tokens, which "
                   "is where the contradiction decision comes from."))
    story.append(P("<b>Running transformers on a small CPU host (PyTorch).</b> RoBERTa-large-MNLI (1.4 GB in "
                   "float32) and BLIP-VQA run locally in PyTorch. To fit the free host's memory, every Linear layer is "
                   "converted with dynamic int8 quantization (torch.ao.quantization.quantize_dynamic): weights are "
                   "stored as 8-bit integers and activations are quantized on the fly, cutting memory about 3x and "
                   "speeding up CPU inference with a small accuracy cost."))
    story.append(P("<b>Image generation.</b> The 'Illustrate' tab draws the agreed answer with FLUX.1-schnell through "
                   "a hosted inference API, as the brief allows for heavy image models. FLUX works in a latent space: a "
                   "<b>VAE</b> encodes images into compact latents and decodes generated latents back to pixels. The "
                   "schnell variant was distilled with latent adversarial diffusion distillation, which adds a "
                   "<b>GAN-style discriminator</b> loss so that a few steps produce a sharp image. Low-agreement answers "
                   "are deliberately not illustrated, so the generator never decorates a hallucination."))

    # ---------------- 7. Results ----------------
    story.append(PageBreak())
    story.append(P("7. Results: 20-question benchmark", "h1"))
    if not have:
        story.append(P("RESULTS PENDING - run the Benchmark tab, download benchmark_results.csv and rebuild this "
                       "report with --csv. Do not submit this draft.", "warn"))
    else:
        thr = float(df["threshold"].iloc[0])
        temp = float(df["temperature"].iloc[0])
        backend = str(df["backend"].iloc[0])
        summary, df = B.metrics(df)
        techs = list(dict.fromkeys(df["technique"]))
        main = "Zero-shot" if "Zero-shot" in techs else techs[0]
        story.append(P(f"Setup: 20 questions (8 easy, 12 long-tail) with known answers, 5 samples each, "
                       f"temperature {temp}, threshold {thr}, model: {X(backend)}. <b>Flagged</b> = agreement below "
                       f"threshold. <b>Actually wrong</b> = the most-agreed answer does not contain the gold answer "
                       f"(checked against accepted aliases)."))
        story.append(P("7.1 Prompt techniques compared", "h2"))
        cols = ["technique", "LLM accuracy", "flagged", "TP", "FP", "FN", "TN", "precision", "recall", "F1",
                "detector acc.", "mean agr. (right)", "mean agr. (wrong)"]
        rows = [["Technique", "LLM acc", "Flag", "TP", "FP", "FN", "TN", "Prec", "Rec", "F1", "Det acc",
                 "Agr R", "Agr W"]]
        rows += [[str(r[c]) for c in cols] for _, r in summary.iterrows()]
        story.append(table(rows, [3.0 * cm] + [(W - 3.0 * cm) / 12] * 12))
        story.append(P("Det acc = detector accuracy (TP+TN)/20. Agr R / Agr W = mean agreement on actually right / wrong answers.", "small"))
        story.append(P("TP = flagged and wrong (caught hallucination), FP = flagged but right (false alarm), "
                       "FN = wrong but not flagged (missed hallucination), TN = right and not flagged.", "small"))
        ms = summary[summary["technique"] == main].iloc[0]
        story.append(P(f"With {main} prompting the LLM answered {ms['LLM accuracy']} questions correctly. The "
                       f"detector flagged {ms['flagged']} questions: {ms['TP']} of them were actually wrong and "
                       f"{ms['FP']} were false alarms; {ms['FN']} wrong answers were missed. Mean agreement was "
                       f"{ms['mean agr. (right)']} on correct answers and {ms['mean agr. (wrong)']} on wrong ones."))

        story.append(P(f"7.2 Flagged vs actually-wrong ({main})", "h2"))
        t = B.flagged_vs_wrong_table(df, main)
        rows = [["#", "Question", "Gold", "Most-agreed answer", "Agr", "Maj", "Flag", "Truth", "Res"]]
        for _, r in t.iterrows():
            rows.append([str(r["id"]), r["question"], r["gold"], str(r["answer"])[:90], f"{r['agreement']:.2f}",
                         r["majority"], r["flagged"], r["actually_wrong"], r["outcome"]])
        story.append(table(rows, [0.8 * cm, W - 11.9 * cm, 2.3 * cm, 3.2 * cm, 1.0 * cm, 0.9 * cm, 1.1 * cm, 1.7 * cm,
                                  0.9 * cm]))
        chart_path = os.path.join(HERE, "results", "report_chart.png")
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        B.chart(df, thr, chart_path)
        story.append(KeepTogether([Spacer(1, 6), img(chart_path, W),
                                   P("Figure 2. Agreement score per question. Crosses are actually-wrong answers; "
                                     "points below the dashed line are flagged.", "small")]))

        # Worked example + failure case
        g = df[df["technique"] == main]
        story.append(P("7.3 Worked example", "h2"))
        ex = g[g["outcome"] == "TP"].sort_values("agreement").head(1)
        if ex.empty:
            ex = g.sort_values("agreement").head(1)
        r = ex.iloc[0]
        story.append(P(f"<b>Q{r['id']}: {X(r['question'])}</b> (gold: {X(r['gold'])})"))
        story.append(table([["Sample", "Answer"]] + [[f"S{i+1}", s] for i, s in enumerate(str(r["samples"]).split(" | "))],
                           [1.6 * cm, W - 1.6 * cm]))
        story.append(P(f"Agreement {r['agreement']:.2f}, majority {r['majority']}, semantic entropy "
                       f"{r['semantic_entropy']:.2f}: result <b>{r['outcome']}</b>."))
        story.append(P("7.4 Failure case", "h2"))
        fail = g[g["outcome"] == "FN"]
        kind = "missed hallucination (FN)"
        if fail.empty:
            fail, kind = g[g["outcome"] == "FP"], "false alarm (FP)"
        if fail.empty:
            story.append(P("No FN or FP occurred in this run. The known weakness still applies: a model that repeats "
                           "the same wrong answer 5 times gets high agreement and would not be flagged."))
        else:
            r = fail.sort_values("agreement", ascending=(kind.startswith("false"))).iloc[0]
            story.append(P(f"<b>Q{r['id']}: {X(r['question'])}</b> - {kind}. Gold: {X(r['gold'])}. Samples: "
                           f"<i>{X(str(r['samples']))}</i>. Agreement {r['agreement']:.2f}."))
            if kind.startswith("missed"):
                story.append(P("The model gave the same wrong answer consistently, so the samples agree and the "
                               "detector cannot tell it from a correct answer. Self-consistency measures the model's "
                               "uncertainty, not truth. Retrieval of a trusted source, or comparing two different "
                               "models, would be needed to catch this kind of error."))
            else:
                story.append(P("The answer was correct but the samples were phrased differently (or included extra "
                               "details), and RoBERTa-MNLI scored some pairs as neutral rather than entailment, which "
                               "lowered agreement. Normalising answers to a short canonical form before NLI would "
                               "reduce such false alarms."))

    # ---------------- 8. Responsible AI ----------------
    story.append(P("8. Responsible AI", "h1"))
    story.append(table([
        ["Requirement", "How it is met"],
        ["Measured hallucination test", "20-question benchmark with flagged vs actually-wrong table and precision / "
                                        "recall (Section 7)."],
        ["Safety guardrail", "Blocks weapon / malware requests and prompt-injection phrases before any model call; "
                             "self-harm mentions get a helpline (Tele-MANAS 14416); 500-character input limit; "
                             "low-agreement answers shown as 'Not sure' and not illustrated."],
        ["Licence / IP note", "Code MIT. gpt-oss: Apache-2.0. RoBERTa-large-MNLI: MIT. BLIP: BSD-3-Clause. "
                              "FLUX.1-schnell: Apache-2.0; generated images are used for education only. "
                              "The Groq API key is stored in Streamlit secrets, not in code."],
        ["Model card", "MODEL_CARD.md in the repository: intended use, method, evaluation, limitations, ethics."],
        ["Bias", "The LLM and NLI model are trained mostly on English, Western text; long-tail questions about other "
                 "regions are more likely to be hallucinated. NLI works only in English."],
        ["Privacy", "No questions or images are stored. Text questions are sent to the Groq API; uploaded images "
                    "stay on the app server (BLIP runs locally); illustration prompts go to the image API."],
    ], [3.6 * cm, W - 3.6 * cm]))

    story.append(P("9. Deployment", "h1"))
    story.append(P(f"The app is a Streamlit app deployed on Streamlit Community Cloud (free tier) from the GitHub repository. It is live at "
                   f"<b>{X(a.app_url)}</b>, and the source code with README and model card is at <b>{X(a.github)}</b>. "
                   "RoBERTa-MNLI is loaded once and cached; BLIP loads on the first image question. Hugging Face Spaces was "
                   "not used because Gradio Spaces now require a paid plan."))

    story.append(P("10. Conclusion and future work", "h1"))
    story.append(P("Asking the same question five times and measuring agreement with an NLI model gives a cheap, "
                   "model-agnostic signal for likely hallucinations, and the app turns it into a clear warning for the "
                   "user. Its main blind spot is consistent errors. Future work: add retrieval of a trusted source "
                   "for a second check, compare answers across two different LLMs, calibrate the threshold on a larger "
                   "question set, and use a multilingual NLI model."))

    story.append(P("References", "h1"))
    for ref in [
        "Manakul, P., Liusie, A., Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for "
        "Generative Large Language Models. EMNLP 2023.",
        "Kuhn, L., Gal, Y., Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty "
        "Estimation in Natural Language Generation. ICLR 2023.",
        "Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. (2024). Detecting hallucinations in large language models using "
        "semantic entropy. Nature 630.",
        "Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.",
        "Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
        "Li, J. et al. (2022). BLIP: Bootstrapping Language-Image Pre-training. ICML 2022.",
        "Sauer, A. et al. (2024). Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation. arXiv:2403.12015.",
    ]:
        story.append(P(ref, "small"))

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="P52 Self-Consistency Hallucination Detector",
                            author=a.name)
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("Saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Name")
    ap.add_argument("--roll", default="52")
    ap.add_argument("--app-url", default="https://<app>.streamlit.app")
    ap.add_argument("--github", default="https://github.com/<user>/<repo>")
    ap.add_argument("--csv", default="results/benchmark_results.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"{a.roll}_{a.name}_P52.pdf"
    build(a)
