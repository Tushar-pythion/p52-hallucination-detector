"""
Look and feel for the Streamlit app: a film-set theme.
Each of the 5 sampled answers is a "take"; the detector checks continuity between takes.
"""
import html

from matplotlib.colors import LinearSegmentedColormap

INK = "#0f1320"       # projection-booth dark
PANEL = "#181d2c"
CREAM = "#ece6d6"     # celluloid
MUTED = "#9d9aa9"
AMBER = "#f2b544"     # projector light
OK = "#4cc38a"
WARN = "#f2b544"
BAD = "#ef5b5b"
GROUP_COLORS = ["#f2b544", "#58c4dc", "#e778a8", "#a48cf0", "#8fd16a"]  # meaning groups A-E

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Antonio:wght@600;700&family=Manrope:wght@400;500;700&display=swap');
.stApp p, .stApp label, .stApp input, .stApp textarea, .stApp li { font-family: 'Manrope', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse 70% 45% at 50% -5%, rgba(242,181,68,.16), transparent 70%),
    radial-gradient(ellipse 120% 90% at 50% 50%, transparent 55%, rgba(0,0,0,.55) 100%),
    #0f1320;
}
h1, h2, h3 { font-family: 'Antonio', 'Arial Narrow', sans-serif !important; letter-spacing: .01em; }
[data-baseweb="tab"] p { font-family: 'Antonio', 'Arial Narrow', sans-serif; font-size: 1.15rem; font-weight: 800; }
section[data-testid="stSidebar"] { border-right: 1px solid #262c3e; }

/* ---------- hero: a projector beam over the title ---------- */
.hero { position: relative; padding: 1.6rem 0 1.2rem; }
.hero .kicker { color: #9d9aa9; font-size: .95rem; margin: 0 0 .2rem; }
.hero h1 {
  font-size: clamp(3.2rem, 8vw, 6.2rem); line-height: .9; margin: 0; padding: 0 !important; font-weight: 900;
  color: #ece6d6; text-shadow: 0 0 34px rgba(242,181,68,.35);
  animation: flicker 1.6s ease-out 1 both;
}
.hero p.lede { max-width: 60ch; color: #cfc9ba; font-size: 1.08rem; line-height: 1.55; margin: .8rem 0 0; }
@keyframes flicker { 0%{opacity:0} 12%{opacity:.9} 18%{opacity:.25} 26%{opacity:1} 32%{opacity:.6} 40%{opacity:1} 100%{opacity:1} }

/* ---------- film strip (the memorable element) ---------- */
.strip {
  --hole: 10px;
  background: #07090f; border-radius: 6px; padding: 26px 14px; margin: .6rem 0 1.2rem; position: relative;
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px;
}
.strip::before, .strip::after {
  content: ""; position: absolute; left: 10px; right: 10px; height: var(--hole);
  background: repeating-linear-gradient(90deg, #2b3042 0 14px, transparent 14px 28px); border-radius: 2px;
}
.strip::before { top: 8px; } .strip::after { bottom: 8px; }
.frame {
  background: linear-gradient(180deg, #1c2233, #141927); border: 2px solid var(--g); border-radius: 4px;
  padding: .7rem .75rem .8rem; min-height: 150px; display: flex; flex-direction: column; gap: .45rem;
  opacity: 0; transform: translateX(-14px); animation: roll .5s ease-out forwards; animation-delay: calc(var(--i) * 140ms);
}
.frame .take { font-family: 'Antonio', 'Arial Narrow', sans-serif; font-weight: 800; font-size: 1.05rem; color: var(--g); display:flex; justify-content: space-between; }
.frame .ans { color: #ece6d6; font-size: .95rem; line-height: 1.35; flex: 1; word-break: break-word; }
.frame .bar { height: 5px; background: #262c3e; border-radius: 3px; overflow: hidden; }
.frame .bar i { display: block; height: 100%; width: var(--c); background: var(--g); }
.frame .meta { color: #9d9aa9; font-size: .78rem; }
.frame.chosen { box-shadow: 0 0 0 1px var(--g), 0 0 22px -4px var(--g); }
@keyframes roll { to { opacity: 1; transform: none; } }

/* ---------- verdict ---------- */
.verdict { display: grid; grid-template-columns: 170px 1fr; gap: 1.6rem; align-items: center; margin: .4rem 0 .6rem; }
.ring { position: relative; width: 170px; height: 170px; }
.ring svg { transform: rotate(-90deg); }
.ring .num { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.ring .num b { font-family: 'Antonio', 'Arial Narrow', sans-serif; font-size: 3.1rem; line-height: 1; color: #ece6d6; }
.ring .num span { display: block; color: #9d9aa9; font-size: .8rem; }
.stamp {
  display: inline-block; font-family: 'Antonio', 'Arial Narrow', sans-serif; font-weight: 900; font-size: 2.3rem;
  line-height: 1; color: var(--v); border: 3px solid var(--v); padding: .25rem .7rem .3rem; border-radius: 6px;
  transform: rotate(-2.5deg); margin-bottom: .7rem;
}
.verdict .why { color: #cfc9ba; max-width: 62ch; line-height: 1.5; margin: 0 0 .6rem; }
.verdict .final { font-size: 1.25rem; color: #ece6d6; }
.verdict .final small { color: #9d9aa9; font-size: .85rem; display: block; }
.stats { display: flex; flex-wrap: wrap; gap: 1.4rem; margin-top: .7rem; color: #9d9aa9; font-size: .88rem; }
.stats b { color: #ece6d6; font-size: 1.05rem; margin-right: .25rem; }

/* ---------- rolling (loading) ---------- */
.rolling { background:#07090f; border-radius:6px; height: 92px; position: relative; overflow: hidden; margin: .6rem 0; }
.rolling::before, .rolling::after {
  content:""; position:absolute; left:0; right:0; height:10px;
  background: repeating-linear-gradient(90deg, #2b3042 0 14px, transparent 14px 28px);
  animation: sprocket .6s linear infinite;
}
.rolling::before{ top:8px } .rolling::after{ bottom:8px }
.rolling span { position:absolute; inset:0; display:grid; place-items:center; color:#f2b544;
  font-family:'Antonio','Arial Narrow',sans-serif; font-weight:800; font-size:1.4rem; }
@keyframes sprocket { from{ background-position: 0 0 } to { background-position: 28px 0 } }

.blocked { border-left: 4px solid #9d9aa9; background:#181d2c; padding: .9rem 1.1rem; border-radius: 4px; color:#ece6d6; }

@media (max-width: 900px) {
  .strip { grid-template-columns: 1fr; }
  .verdict { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .hero h1, .frame, .rolling::before, .rolling::after { animation: none !important; opacity: 1; transform: none; }
}
</style>
"""

HERO = (
    "<div class='hero'>"
    "<p class='kicker'>P52 · Self-Consistency Hallucination Detector</p>"
    "<h1>Five Takes</h1>"
    "<p class='lede'>Ask the AI the same question five times. If it really knows the answer, every take matches. "
    "If the takes drift apart, it is probably making things up.</p>"
    "</div>"
)

ROLLING = "<div class='rolling'><span>Rolling five takes…</span></div>"


def heat_cmap():
    return LinearSegmentedColormap.from_list("booth", ["#1a2030", "#6b5a2e", AMBER])


def blocked(msg):
    return f"<div class='blocked'>🛡️ <b>Guardrail.</b> {html.escape(msg)}</div>"


def filmstrip(r):
    group_of = {i: g for g, members in enumerate(r["clusters"]) for i in members}
    frames = []
    for i, ans in enumerate(r["answers"]):
        g = group_of[i]
        color = GROUP_COLORS[g % len(GROUP_COLORS)]
        conf = float(r["confidence"][i])
        chosen = " chosen" if i == r["rep_index"] else ""
        star = "★ most agreed" if chosen else ""
        frames.append(
            f"<div class='frame{chosen}' style='--g:{color};--i:{i};--c:{conf * 100:.0f}%'>"
            f"<div class='take'><span>Take {i + 1}</span><span>{chr(65 + g)}</span></div>"
            f"<div class='ans'>{html.escape(ans)}</div>"
            f"<div class='bar'><i></i></div>"
            f"<div class='meta'>confidence {conf:.2f} · meaning {chr(65 + g)} {star}</div>"
            f"</div>")
    return "<div class='strip'>" + "".join(frames) + "</div>"


def verdict(r):
    agr = r["agreement"]
    if r["level"] == "low":
        color, stamp = BAD, "Continuity broken"
        why = ("The takes don't match. The model is guessing, so this answer is <b>likely a hallucination</b>. "
               "Check a reliable source before using it.")
        final = "Not sure"
    elif r["level"] == "medium":
        color, stamp = WARN, "Takes mostly match"
        why = "Most takes agree, but not all. Double-check important details."
        final = html.escape(r["answer"])
    else:
        color, stamp = OK, "Continuity holds"
        why = "Every take tells the same story. That means the model is consistent, which is not proof it is right."
        final = html.escape(r["answer"])
    circ = 2 * 3.14159 * 70
    ring = (f"<div class='ring'><svg width='170' height='170' viewBox='0 0 170 170'>"
            f"<circle cx='85' cy='85' r='70' fill='none' stroke='#262c3e' stroke-width='12'/>"
            f"<circle cx='85' cy='85' r='70' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' "
            f"stroke-dasharray='{circ * agr:.1f} {circ:.1f}'/></svg>"
            f"<div class='num'><div><b>{agr * 100:.0f}%</b><span>agreement</span></div></div></div>")
    stats = (f"<div class='stats'><span><b>{round(r['majority_frac'] * 5)}/5</b>takes in the biggest group</span>"
             f"<span><b>{len(r['clusters'])}</b>different meanings</span>"
             f"<span><b>{r['semantic_entropy']:.2f}</b>semantic entropy</span>"
             f"<span><b>{r['latency']:.1f}s</b>{html.escape(r['backend'])}, {html.escape(r['technique'])}, "
             f"T={r['temperature']}</span></div>")
    return (f"<div class='verdict'>{ring}<div><div class='stamp' style='--v:{color}'>{stamp}</div>"
            f"<p class='why'>{why}</p><div class='final'><small>Answer</small>{final}</div>{stats}</div></div>")
