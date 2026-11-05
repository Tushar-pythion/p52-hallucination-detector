"""
P52 · Self-Consistency Hallucination Detector - Streamlit app.
Deploy free on Streamlit Community Cloud. Put GROQ_API_KEY in the app's Secrets.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="P52 · Hallucination Detector", page_icon="🔍", layout="wide")

# Secrets (Streamlit Cloud -> Settings -> Secrets) become environment variables
try:
    for key in ("GROQ_API_KEY", "HF_TOKEN"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import detector as D  # noqa: E402
import benchmark as B  # noqa: E402
import ui  # noqa: E402

# Dark "projection booth" chart theme
D.C_INK, D.C_INK2, D.C_GRID, D.C_SURF = ui.CREAM, ui.MUTED, "#2a3044", ui.INK
D.C_GOOD, D.C_WARN, D.C_BAD = ui.OK, ui.WARN, ui.BAD
D.HEAT_CMAP, D.HEAT_TEXT_HIGH, D.HEAT_TEXT_LOW = ui.heat_cmap(), ui.INK, ui.CREAM
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading RoBERTa-MNLI (first time only, about 1 minute)...")
def load_nli():
    D._nli()
    return True


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    technique = st.radio("U1 · Prompt technique", D.TECHNIQUES, index=0)
    temperature = st.slider("U1 · Temperature", 0.0, 1.5, 0.9, 0.1)
    backend = st.selectbox("U2 · LLM", list(D.BACKENDS))
    threshold = st.slider("U5 · Warning threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption("5 answers are sampled for every question.")
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it in Settings → Secrets.")

st.markdown(ui.HERO, unsafe_allow_html=True)

load_nli()
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Detector", "Why? (attention)", "Illustrate",
                                        "Benchmark (20 questions)", "About"])

# ---------------- Detector ----------------
with tab1:
    examples = ["", "What is the capital of Australia?", "Who was the first President of Botswana?",
                "What is the atomic number of rhenium?", "Who won the 1987 Nobel Prize in Chemistry?"]
    ex = st.selectbox("Example questions (optional)", examples)
    question = st.text_area("Question", value=ex, height=80,
                            placeholder="e.g. Who was the first President of Botswana?")
    up = st.file_uploader("Optional image (image questions are answered by BLIP)", type=["png", "jpg", "jpeg", "webp"])
    if up:
        st.image(up, width=220)
    if st.button("Ask 5 times", type="primary"):
        image = Image.open(up) if up else None
        reel = st.empty()
        reel.markdown(ui.ROLLING, unsafe_allow_html=True)
        try:
            st.session_state["res"] = D.detect(question, technique, temperature, backend,
                                               image=image, threshold=threshold)
        except Exception as e:
            st.session_state["res"] = None
            st.error(f"Error: {e}")
        reel.empty()

    r = st.session_state.get("res")
    if r and r.get("blocked"):
        st.markdown(ui.blocked(r["message"]), unsafe_allow_html=True)
    elif r:
        st.markdown(ui.verdict(r), unsafe_allow_html=True)
        st.markdown("### The five takes")
        st.markdown(ui.filmstrip(r), unsafe_allow_html=True)
        st.caption("Takes with the same letter mean the same thing (RoBERTa-MNLI entailment ≥ 0.5). "
                   "The bar is each take's confidence: its average agreement with the other four.")
        with st.expander("U4 · Confidence chart and agreement heatmap", expanded=True):
            show_fig(D.confidence_chart(r))

# ---------------- Attention ----------------
with tab2:
    st.write("How RoBERTa-MNLI compares two sampled answers. Brighter cells = more attention from a token of "
             "answer B to a token of answer A (last layer, averaged over 16 heads).")
    r = st.session_state.get("res")
    if not r or r.get("blocked"):
        st.info("Run the detector first.")
    else:
        names = [f"S{i}" for i in range(1, len(r["answers"]) + 1)]
        a = st.selectbox("Answer A", names, index=0)
        b = st.selectbox("Answer B", names, index=1)
        i, j = names.index(a), names.index(b)
        st.markdown(f"**A ({a}):** {r['answers'][i]}  \n**B ({b}):** {r['answers'][j]}")
        if st.button("Show attention"):
            with st.spinner("Computing attention..."):
                show_fig(D.attention_figure(r["answers"][i], r["answers"][j]))

# ---------------- Illustrate ----------------
with tab3:
    st.write("Draws the agreed answer with **FLUX.1-schnell** (a diffusion model) through a hosted image API. "
             "Low-agreement answers are not illustrated.")
    r = st.session_state.get("res")
    seed = st.number_input("Seed", value=0, step=1)
    if st.button("Generate illustration"):
        if not r or r.get("blocked"):
            st.info("Run the detector first.")
        elif r["level"] == "low":
            st.warning("⚠️ Not illustrating: this answer has low agreement and may be a hallucination.")
        else:
            with st.spinner("Generating image (10-60 s)..."):
                try:
                    img, src = D.illustrate(f"{r['question']} {r['answer']}", seed)
                    st.image(img, caption=f"{r['answer']} · {src}")
                except Exception as e:
                    st.error(str(e))

# ---------------- Benchmark ----------------
with tab4:
    st.write("Runs the detector on 20 questions with known answers and compares **flagged** vs **actually "
             "wrong**. Uses the temperature, LLM and threshold from the sidebar. One technique ≈ 100 LLM calls "
             "(2-5 minutes).")
    techs = st.multiselect("Techniques to compare", D.TECHNIQUES, default=D.TECHNIQUES)
    if st.button("Run benchmark", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            df = B.run(techs or ["Zero-shot"], temperature, backend, threshold,
                       progress=lambda f, desc="": bar.progress(min(float(f), 1.0), text=desc))
            st.session_state["bench"] = df
            bar.progress(1.0, text="Done")
        except Exception as e:
            st.error(f"Benchmark stopped: {e}. Wait a minute and try again with one technique.")
    df = st.session_state.get("bench")
    if df is not None:
        summary, df = B.metrics(df)
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(),
                           "benchmark_results.csv", "text/csv", type="primary")
        st.markdown("**Summary** - TP = flagged & wrong, FP = flagged & right, FN = missed hallucination")
        st.dataframe(summary, hide_index=True)
        st.markdown("**Flagged vs actually-wrong**")
        t = B.flagged_vs_wrong_table(df)
        t.insert(0, "technique", df["technique"].values)
        st.dataframe(t, hide_index=True)
        show_fig(B.chart(df, float(df["threshold"].iloc[0])))

# ---------------- About ----------------
with tab5:
    if os.path.exists("ABOUT.md"):
        st.markdown(open("ABOUT.md", encoding="utf-8").read())
    if os.path.exists("docs/lifecycle.png"):
        st.image("docs/lifecycle.png", caption="Generative-AI lifecycle of this project")
