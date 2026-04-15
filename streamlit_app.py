"""
AutoDS  ·  Conversational Pipeline Dashboard
Upload data → run pipeline → chat with results in plain language
"""

from __future__ import annotations
import json, os, re, sys, traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env (must happen before any agent imports so API keys are visible) ──
try:
    from utils import load_project_env
    load_project_env(__file__)
except Exception:
    pass

BASE = PROJECT_ROOT / "autods_pipeline_output"

# ── Brand tokens ──────────────────────────────────────────────────────────────
NAVY   = "#0E367C"
BLUE   = "#2361AE"
BRIGHT = "#2B7CC5"
CYAN   = "#3BA5C7"
MIST   = "#9CB7C8"
GREEN  = "#1B8A56"
ORANGE = "#D97706"
BG     = "#FBFAF7"
BORDER = "#DDE4EC"
SUB    = "#6E7D8F"
WHITE  = "#FFFFFF"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoDS · AI Data Scientist",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap');

html,body,[class*="css"]{{
  font-family:'Inter','Noto Sans SC',system-ui,sans-serif;
  background:{BG};color:#24364B;
}}

/* Sidebar */
section[data-testid="stSidebar"]>div{{background:{NAVY};}}
section[data-testid="stSidebar"] *  {{color:rgba(255,255,255,.85)!important;}}
section[data-testid="stSidebar"] hr{{border-color:rgba(255,255,255,.12)!important;}}
section[data-testid="stSidebar"] .stRadio label{{font-size:13px!important;}}

/* Metric cards */
[data-testid="metric-container"]{{
  background:#fff;border:1px solid {BORDER};border-radius:10px;
  padding:16px 20px 12px;box-shadow:0 1px 4px rgba(14,54,124,.07);
}}
[data-testid="metric-container"] label{{
  font-size:10px!important;font-weight:700!important;
  text-transform:uppercase;letter-spacing:.07em;color:{SUB}!important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{
  font-size:24px!important;font-weight:800!important;color:{NAVY}!important;
}}

h2{{color:{NAVY}!important;font-weight:800!important;letter-spacing:-.3px;}}
h3{{color:#0E2F63!important;font-weight:700!important;font-size:15px!important;}}
hr{{border-color:{BORDER}!important;margin:8px 0 18px;}}

/* Chat bubbles */
.msg-user{{
  background:{NAVY};color:#fff;border-radius:18px 18px 4px 18px;
  padding:12px 16px;margin:0 0 4px auto;max-width:75%;
  font-size:14px;line-height:1.55;width:fit-content;
}}
.msg-bot{{
  background:#fff;color:#24364B;border-radius:18px 18px 18px 4px;
  border:1px solid {BORDER};padding:14px 18px;margin:0 auto 4px 0;
  max-width:88%;font-size:13.5px;line-height:1.65;width:fit-content;
  box-shadow:0 1px 3px rgba(14,54,124,.06);
}}
.msg-bot b{{color:{NAVY};}}
.msg-bot code{{background:#EAF3FB;color:{BLUE};padding:1px 5px;border-radius:3px;font-size:12px;}}
.msg-system{{
  text-align:center;font-size:11px;color:{SUB};letter-spacing:.03em;
  padding:6px 0;
}}

/* Suggestion pills */
.pill{{
  display:inline-block;background:#EAF3FB;color:{BLUE};
  border:1px solid #B8D8F0;border-radius:20px;
  padding:5px 13px;font-size:12px;font-weight:600;
  cursor:pointer;margin:3px;
}}

/* Buttons */
.stButton>button{{
  background:{NAVY};color:#fff;border:none;border-radius:8px;
  font-weight:700;font-size:13px;padding:9px 22px;
}}
.stButton>button:hover{{background:{BLUE};}}

/* Detail cards */
.det-row{{
  display:flex;gap:10px;padding:10px 0;
  border-bottom:1px solid {BORDER};align-items:flex-start;
}}
.det-dot{{
  width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:7px;
}}

/* Tab-style toggles */
.tab-btn{{
  border:1px solid {BORDER};background:#fff;color:{SUB};
  border-radius:6px;padding:5px 14px;font-size:12px;font-weight:600;cursor:pointer;
}}
.tab-btn.active{{background:{NAVY};color:#fff;border-color:{NAVY};}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_json(rel: str) -> dict:
    p = BASE / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def load_csv(rel: str) -> pd.DataFrame:
    p = BASE / rel
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def fmt(v: Any, digits: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v) if v is not None else "—"

def pct(v: Any) -> str:
    return f"{float(v)*100:.1f}%" if v is not None else "—"

def score_label(v: float) -> tuple[str, str]:
    """Return (label, colour) for a 0-1 metric score."""
    if v >= 0.95: return "Excellent",  GREEN
    if v >= 0.85: return "Good",        BLUE
    if v >= 0.70: return "Acceptable",  ORANGE
    return "Needs Work", "#C0392B"

def friendly_metric(metric_key: str, value: float, problem_type: str) -> str:
    label, _ = score_label(value)
    if problem_type == "regression":
        msgs = {
            "rmse":  f"Prediction error (RMSE) is {value:.3f}, representing the average deviation between predicted and actual values.",
            "mae":   f"Mean Absolute Error (MAE) is {value:.3f}.",
            "r2":    f"The model explains {value*100:.1f}% of the variance in the target variable (R²={value:.3f}).",
        }
        return msgs.get(metric_key, f"{metric_key.upper()} = {value:.3f}")
    else:
        quality = ["reasonably well","well","excellently"][min(2,int((value-0.5)/0.175))]
        msgs = {
            "roc_auc":  f"Model ROC-AUC={value:.3f} ({label}) — discriminates between classes {quality}.",
            "f1":       f"F1={value:.3f} ({label}), balancing precision and recall.",
            "accuracy": f"Accuracy {value*100:.1f}% ({label}).",
            "precision":f"Precision {value*100:.1f}% ({label}) — reliability when predicting positive cases.",
            "recall":   f"Recall {value*100:.1f}% ({label}) — proportion of actual positives detected.",
        }
        return msgs.get(metric_key, f"{metric_key.upper()} = {value:.3f}")


def run_pipeline(csv_path: str, desc: str, target: str, prob: str) -> tuple[bool, str]:
    try:
        from autods_implementation_guide import PipelineConfig, DataSciencePipeline
        cfg = PipelineConfig()
        cfg.data_path = csv_path
        cfg.business_description = desc
        cfg.target_column = target if target else None
        cfg.problem_type = prob
        cfg.use_planner = True
        DataSciencePipeline(cfg).run_complete_pipeline()
        return True, ""
    except Exception:
        return False, traceback.format_exc()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PIPELINE DATA (cached per run)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_pipeline_data() -> dict:
    understanding  = load_json("01_understanding/data_understanding_summary.json")
    cleaning       = load_json("02_cleaning/cleaning_report.json")
    feat_meta      = load_json("03_feature_engineering/feature_metadata.json")
    modelling_meta = load_json("04_modelling/modelling_metadata.json")
    best_metrics   = load_json("04_modelling/best_model_metrics.json")
    eval_summary   = load_json("05_evaluation/evaluation_summary.json")
    report         = load_json("06_reports/report.json")
    report_input   = load_json("06_reports/pipeline_report_input.json")
    leaderboard    = load_csv("04_modelling/leaderboard.csv")
    feat_imp       = load_csv("04_modelling/best_model_feature_importance.csv")

    mod_d          = report_input.get("modeling", {})
    best_model_name= (mod_d.get("best_model_name")
                      or mod_d.get("best_model", {}).get("name")
                      or eval_summary.get("best_model_name", "—"))
    primary_metric = (eval_summary.get("primary_metric")
                      or mod_d.get("primary_metric", "roc_auc"))
    # lb_list: evaluation_summary.comparison_table uses "model_name" key (matches frontend);
    # fall back to pipeline_report_input models_compared if needed
    lb_list        = (eval_summary.get("benchmark_overview", {}).get("comparison_table")
                      or mod_d.get("leaderboard")
                      or mod_d.get("models_compared", []))
    feat_eng       = report_input.get("feature_engineering", {})
    final_feat_cnt = feat_eng.get("final_feature_count", len(feat_imp))

    clean_sum   = cleaning.get("cleaning_summary", {})
    # cleaning_report.json: input_data/output_data are flat dicts {rows, columns}, no "shape" nesting
    in_shape    = cleaning.get("input_data",  {})
    out_shape   = cleaning.get("output_data", {})
    n_rows_in   = in_shape.get("rows", "—")
    n_rows_out  = out_shape.get("rows", "—")

    bm = {}
    if best_metrics:
        bm = best_metrics
    elif lb_list:
        bm = lb_list[0]

    yt = load_csv("03_feature_engineering/y_train.csv")
    yv = load_csv("03_feature_engineering/y_test.csv")

    dh  = understanding.get("downstream_handoff", {})
    mo  = dh.get("modelling_agent", {})

    return dict(
        understanding=understanding,
        cleaning=cleaning,
        modelling_meta=modelling_meta,
        best_metrics=best_metrics,
        eval_summary=eval_summary,
        report=report,
        report_input=report_input,
        leaderboard=leaderboard,
        feat_imp=feat_imp,
        lb_list=lb_list,
        best_model_name=best_model_name,
        primary_metric=primary_metric,
        final_feat_cnt=final_feat_cnt,
        clean_sum=clean_sum,
        in_shape=in_shape,
        out_shape=out_shape,
        n_rows_in=n_rows_in,
        n_rows_out=n_rows_out,
        bm=bm,
        n_train=len(yt), n_test=len(yv),
        major_findings=understanding.get("major_findings",[]),
        primary_risks=understanding.get("primary_risks",[]),
        next_steps=understanding.get("recommended_next_steps",[]),
        exec_summary=understanding.get("executive_summary",""),
        problem_type=mo.get("problem_type","classification"),
        target_column=mo.get("target_column","—"),
        business_report=report.get("business_report", report.get("business", "")),
        timestamp=report_input.get("meta",{}).get("timestamp","—"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  (cached per Streamlit process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_llm():
    """Return the shared LLM client, or None when no API key is available."""
    try:
        from utils import build_chat_llm
        return build_chat_llm()
    except Exception:
        return None


def _llm_answer(question: str, d: dict) -> str:
    """LLM fallback for questions the rule engine doesn't recognise."""
    llm = _get_llm()
    if llm is None:
        return (
            "I can answer the following questions — click or type one:\n\n"
            "• What does this dataset look like?\n"
            "• How good is the model?\n"
            "• What are the risks?\n"
            "• Which features matter most?\n"
            "• What should I do next?\n"
            "• Is the model ready to deploy?"
        )
    pm = d["primary_metric"]
    metric_val = d["bm"].get(f"test_{pm}", d["bm"].get("test_roc_auc", 0)) or 0
    context = (
        f"AutoDS automated ML pipeline results:\n"
        f"- Task: {d['problem_type']} · Target column: {d['target_column']}\n"
        f"- Best model: {d['best_model_name'].replace('_', ' ').title()} · "
        f"{pm.upper()}={metric_val:.3f}\n"
        f"- Dataset: {d['n_rows_in']} rows → {d['n_rows_out']} rows after cleaning\n"
        f"- Final features: {d['final_feat_cnt']}\n"
        f"- Key findings: {'; '.join(str(f) for f in d['major_findings'][:4])}\n"
        f"- Risk flags: {'; '.join(str(r) for r in d['primary_risks'][:3]) or 'None detected'}\n"
        f"- Recommended next steps: {'; '.join(str(s) for s in d['next_steps'][:3])}"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = llm.invoke([
            SystemMessage(content=(
                "You are a concise, expert data science advisor embedded in an automated ML platform. "
                "The user has just run a full pipeline on their dataset and is asking questions about the results. "
                "Answer in 2-4 sentences using plain language suitable for a business audience. "
                "Never invent numbers — only use the data provided below.\n\n"
                + context
            )),
            HumanMessage(content=question),
        ])
        return str(getattr(resp, "content", resp))
    except Exception as exc:
        return (
            f"I couldn't process that question right now ({type(exc).__name__}). "
            "Try: How good is the model? What are the key risks? Which features matter most?"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ANSWER ENGINE  (rule-based fast-path; LLM fallback for unknown queries)
# ─────────────────────────────────────────────────────────────────────────────
GREETINGS = {"hi","hello","hey","start","help","greetings"}

def _answer(question: str, d: dict) -> str:
    q   = question.lower().strip()
    bm  = d["bm"]
    pm  = d["primary_metric"]
    pt  = d["problem_type"]
    bmn = d["best_model_name"]
    tgt = d["target_column"]
    roc = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
    f1  = bm.get("test_f1", 0)
    acc = bm.get("test_accuracy", 0)

    # ── Opening greeting ──────────────────────────────────────────────────────
    if any(w in q for w in GREETINGS):
        return _opening_msg(d)

    # ── Data overview ─────────────────────────────────────────────────────────
    if any(w in q for w in ["dataset","data","overview","summary","about"]):
        es = d["exec_summary"]
        if es:
            findings = "\n".join(f"• {f}" for f in d["major_findings"][:4])
            return f"{es}\n\n**Key findings:**\n{findings}"
        return "No data understanding results yet. Please run the pipeline first."

    # ── Risks ─────────────────────────────────────────────────────────────────
    if any(w in q for w in ["risk","issue","danger","warning","problem"]):
        risks = d["primary_risks"]
        if not risks:
            return "No significant data risks detected. Overall data quality is good."
        lines = "\n".join(f"• {r}" for r in risks)
        return f"The following potential risks were detected:\n\n{lines}\n\nReview each before deploying."

    # ── Next steps ────────────────────────────────────────────────────────────
    if any(w in q for w in ["next","recommend","action","what should","steps","todo"]):
        steps = d["next_steps"]
        if not steps:
            return "Pipeline complete. Consider deploying the model to a staging environment and monitoring performance."
        lines = "\n".join(f"{i+1}. {s}" for i,s in enumerate(steps))
        br = d.get("business_report","")
        rec_section = ""
        if "recommend" in br.lower() or "action" in br.lower():
            for line in br.split("\n"):
                if any(k in line.lower() for k in ["recommend","action","next"]):
                    rec_section = line.strip()
                    break
        return (
            f"Based on the analysis, here are the recommended steps:\n\n{lines}"
            + (f"\n\n{rec_section}" if rec_section else "")
        )

    # ── Model result ──────────────────────────────────────────────────────────
    if any(w in q for w in ["model","result","accurate","performance","score","how good"]):
        score_str = friendly_metric(pm, roc, pt)
        lbl, col = score_label(roc)
        imb_flag = ""
        for f in d["major_findings"]:
            if "imbalance" in f.lower():
                imb_flag = "\n\n⚠ Note: Class imbalance detected. The model may favour the majority class — consider F1 or Recall as primary metrics."
                break
        return (
            f"**Best model: {bmn.replace('_',' ').title()}**  ·  Overall rating: **{lbl}**\n\n"
            f"{score_str}\n"
            f"F1={fmt(f1)}  ·  Accuracy={pct(acc)}\n"
            f"{imb_flag}\n\n"
            f"{len(d['lb_list'])} candidate models were compared. {bmn.replace('_',' ').title()} achieved the best overall performance."
        )

    # ── Deploy ────────────────────────────────────────────────────────────────
    if any(w in q for w in ["deploy","production","release","ship","ready","can we use"]):
        lbl, _ = score_label(roc)
        if roc >= 0.85:
            verdict = "✅ Model performance is strong. Consider deploying to a staging environment for further validation."
        elif roc >= 0.70:
            verdict = "⚠ Model performance is moderate. Collect more data before considering production deployment."
        else:
            verdict = "❌ Model performance is insufficient for production. Optimise data or model first."
        return (
            f"{verdict}\n\n"
            "**Pre-deployment checklist:**\n"
            "1. Have domain experts validate the business logic of predictions\n"
            "2. Set up model performance monitoring (data drift alerts)\n"
            "3. Prepare a rollback plan\n"
            "4. Confirm target column semantics align with business intent"
        )

    # ── Feature importance ────────────────────────────────────────────────────
    if any(w in q for w in ["feature","variable","important","which","top"]):
        fi = d["feat_imp"]
        if fi.empty:
            return "No feature importance data available."
        top3 = fi.head(3)
        lines = "\n".join(
            f"{i+1}. **{row['feature_name']}** (importance score: {int(row['importance'])})"
            for i, row in top3.iterrows()
        )
        return (
            f"The three features with the greatest impact on predictions are:\n\n{lines}\n\n"
            "These variables are the primary source of predictive power. If they are hard to obtain in production, "
            "the model's deployability should be re-evaluated."
        )

    # ── Problem type ──────────────────────────────────────────────────────────
    if any(w in q for w in ["why","regression","classification","task","infer","type"]):
        if pt == "regression":
            return (
                f"The target column **{tgt}** is numeric with many distinct values (continuous), "
                "so the system identified this as a **regression task** (predicting a continuous value).\n\n"
                "If you believe it should be a classification task, specify the problem type explicitly when submitting."
            )
        else:
            return (
                f"The target column **{tgt}** was identified as a **classification task** "
                "because its values are discrete categories (e.g. yes/no, class labels).\n\n"
                "If you believe it should be regression, specify the problem type explicitly when submitting."
            )

    # ── Cleaning ──────────────────────────────────────────────────────────────
    if any(w in q for w in ["clean","missing","dirty","preprocess","impute"]):
        cs = d["clean_sum"]
        rows_removed = cs.get("rows_removed", 0)
        retention    = cs.get("data_retention_pct", cs.get("data_retention_percentage", 0))
        n_in  = d["n_rows_in"]
        n_out = d["n_rows_out"]
        return (
            f"Data cleaning complete.\n\n"
            f"• Input: {n_in} rows\n"
            f"• Output: {n_out} rows\n"
            f"• Removed {rows_removed} anomalous/duplicate rows\n"
            f"• Data retention: {retention:.1f}%\n\n"
            f"{'✅ Data quality is good — retention above 95%.' if float(retention or 0) >= 95 else '⚠ Retention is low — review raw data quality.'}"
        )

    # ── Business report ───────────────────────────────────────────────────────
    if any(w in q for w in ["report","summary","conclusion","business","executive"]):
        br = d.get("business_report","")
        if br:
            lines = [l.strip() for l in br.split("\n") if l.strip() and not l.startswith("#")]
            snippet = " ".join(lines[:6])[:600]
            return f'**Business Report Summary:**\n\n{snippet}\n\n(Switch to the **Technical Dashboard** for the full technical report.)'
        return "No business report available. Please run the pipeline first."

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if any(w in q for w in ["leaderboard","compare","all models","candidate","ranking"]):
        lb = d["lb_list"]
        if not lb:
            return "No leaderboard data available."
        lines = []
        for i, m in enumerate(lb[:5]):
            lines.append(
                f"{i+1}. **{m['model_name'].replace('_',' ').title()}** — "
                f"ROC-AUC {m.get('test_roc_auc',0):.3f}  F1 {m.get('test_f1',0):.3f}"
            )
        return "**Model Leaderboard (sorted by ROC-AUC):**\n\n" + "\n".join(lines)

    # ── Fallback → LLM ────────────────────────────────────────────────────────
    return _llm_answer(question, d)


def _opening_msg(d: dict) -> str:
    pt  = d["problem_type"]
    tgt = d["target_column"]
    bm  = d["bm"]
    pm  = d["primary_metric"]
    roc = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
    lbl, _ = score_label(roc)
    risks = d["primary_risks"]
    risk_hint = f"Data has {len(risks)} potential risks that require attention." if risks else "Overall data quality is good."
    pt_label = "Regression (predict continuous values)" if pt == "regression" else "Classification (predict category labels)"
    return (
        f"Hello 👋 I have completed the full-pipeline analysis of your dataset.\n\n"
        f"**Task type:** {pt_label}\n"
        f"**Target column:** `{tgt}`\n"
        f"**Best model performance:** {lbl}  ({pm.upper()} = {roc:.3f})\n"
        f"**Data status:** {risk_hint}\n\n"
        "You can ask me:\n"
        "• What issues does this dataset have?\n"
        "• How accurate is the model?\n"
        "• Which variables matter most?\n"
        "• Is the model ready to deploy?\n"
        "• What should I do next?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role":"user"|"bot", "text":str}
if "detail_level" not in st.session_state:
    st.session_state.detail_level = "business"   # business | analyst | technical
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding:10px 0 18px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <div style="background:rgba(255,255,255,.12);border-radius:8px;width:36px;height:36px;
                    display:flex;align-items:center;justify-content:center;">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <circle cx="4"  cy="18" r="2.5" fill="#2361AE"/>
            <circle cx="12" cy="10" r="2.5" fill="#2B7CC5"/>
            <circle cx="20" cy="5"  r="2.5" fill="#3BA5C7"/>
            <path d="M6.5 17 Q10 12 9.8 10" stroke="#2361AE" stroke-width="1.6" fill="none" stroke-linecap="round"/>
            <path d="M14.2 9.5 Q17 6.5 17.5 5" stroke="#3BA5C7" stroke-width="1.6" fill="none" stroke-linecap="round"/>
            <path d="M20 7.5 Q21 13 17 16 Q13 19 9 18 Q6 17.5 5 20" stroke="#9CB7C8" stroke-width="1.3"
                  fill="none" stroke-linecap="round" stroke-dasharray="2 2"/>
          </svg>
        </div>
        <div>
          <div style="font-size:17px;font-weight:800;color:#fff;letter-spacing:-.2px;">AUTODS</div>
          <div style="font-size:10px;color:rgba(255,255,255,.45);letter-spacing:.05em;">Auto Data Science</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Main navigation
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;'
                'color:rgba(255,255,255,.35);padding:2px 0 8px;">Navigation</div>', unsafe_allow_html=True)
    view = st.radio("View", ["💬  Chat Analysis", "🚀  Submit New Job", "📊  Technical Dashboard"],
                    label_visibility="collapsed")

    results_exist = (BASE / "06_reports" / "pipeline_report_input.json").exists()

    if results_exist:
        st.markdown("---")
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;'
                    'color:rgba(255,255,255,.35);padding:2px 0 8px;">Answer Depth</div>', unsafe_allow_html=True)
        level = st.radio("Level", ["🟢  Business", "🟡  Analyst", "🔵  Technical"],
                         label_visibility="collapsed",
                         index=["🟢  Business","🟡  Analyst","🔵  Technical"].index(
                             {"business":"🟢  Business","analyst":"🟡  Analyst","technical":"🔵  Technical"}
                             .get(st.session_state.detail_level,"🟢  Business")))
        st.session_state.detail_level = {
            "🟢  Business":"business","🟡  Analyst":"analyst","🔵  Technical":"technical"
        }[level]
        st.markdown("---")
        if st.button("🗑  Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW — CHAT (default)
# ─────────────────────────────────────────────────────────────────────────────
if "Chat" in view:

    results_exist = (BASE / "06_reports" / "pipeline_report_input.json").exists()

    if not results_exist:
        st.markdown(f"""
        <div style="text-align:center;padding:80px 0;">
          <div style="font-size:40px;margin-bottom:16px;">📭</div>
          <div style="font-size:20px;font-weight:700;color:{NAVY};margin-bottom:8px;">No Analysis Results Yet</div>
          <div style="font-size:14px;color:{SUB};">Please switch to <strong>Submit New Job</strong> to upload data and run the pipeline.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    d = get_pipeline_data()

    # Auto-send opening message on first load
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role":"bot","text":_opening_msg(d)})

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding-bottom:16px;border-bottom:1px solid {BORDER};margin-bottom:20px;">
      <div>
        <div style="font-size:21px;font-weight:800;color:{NAVY};letter-spacing:-.4px;">AI Data Advisor</div>
        <div style="font-size:12px;color:{SUB};margin-top:3px;">
          Answering your questions about the analysis results in plain language
        </div>
      </div>
      <div style="font-size:11px;color:{SUB};text-align:right;line-height:1.7;">
        Target <b style="color:{BLUE};">{d["target_column"]}</b><br>
        Best Model <b style="color:{BLUE};">{d["best_model_name"].replace("_"," ").title()}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detail-level context banner ───────────────────────────────────────────
    level_labels = {"business":"Business · Plain Language","analyst":"Analyst · Key Metrics","technical":"Technical · Full Details"}
    level_colors = {"business":GREEN,"analyst":ORANGE,"technical":BLUE}
    lv = st.session_state.detail_level
    st.markdown(f"""
    <div style="background:{level_colors[lv]}18;border:1px solid {level_colors[lv]}44;
                border-radius:6px;padding:6px 14px;margin-bottom:16px;font-size:12px;color:{level_colors[lv]};font-weight:600;">
      Current mode: {level_labels[lv]}
    </div>""", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:10px;">'
                            f'<div class="msg-user">{msg["text"]}</div></div>', unsafe_allow_html=True)
            else:
                # Render markdown properly for bot messages
                text = msg["text"]
                # Convert **bold** and bullet points for HTML
                text_html = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                text_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_html)
                text_html = re.sub(r'`(.+?)`', r'<code>\1</code>', text_html)
                text_html = text_html.replace("\n\n","<br><br>").replace("\n","<br>")
                st.markdown(f'<div style="display:flex;margin-bottom:10px;">'
                            f'<div class="msg-bot">{text_html}</div></div>', unsafe_allow_html=True)

    # ── Suggestion pills ──────────────────────────────────────────────────────
    suggestions = [
        "What issues does this dataset have?",
        "How is the model performing?",
        "Which variables matter most?",
        "Is the model ready to deploy?",
        "What should I do next?",
        "Why classification/regression?",
        "How was the data cleaned?",
        "Model leaderboard",
    ]
    st.markdown("**Quick questions:**")
    pill_cols = st.columns(4)
    for i, sug in enumerate(suggestions):
        with pill_cols[i % 4]:
            if st.button(sug, key=f"pill_{i}"):
                st.session_state.chat_history.append({"role":"user","text":sug})
                answer = _answer(sug, d)
                # Add analyst/technical detail
                if st.session_state.detail_level == "analyst":
                    bm = d["bm"]; pm = d["primary_metric"]
                    roc = bm.get("test_roc_auc") or bm.get(f"test_{pm}",0)
                    answer += (f"\n\n📊 **Additional metrics:** {pm.upper()}={roc:.3f}  "
                               f"F1={bm.get('test_f1',0):.3f}  Acc={bm.get('test_accuracy',0):.3f}")
                elif st.session_state.detail_level == "technical":
                    answer += f"\n\n🔬 For full technical details, switch to the **Technical Dashboard**."
                st.session_state.chat_history.append({"role":"bot","text":answer})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Free-text input ───────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input("Type your question…", label_visibility="collapsed",
                                       placeholder="e.g. How accurate is the model? What are the risks?")
        with col_btn:
            submitted = st.form_submit_button("Send ↵")

    if submitted and user_input.strip():
        q = user_input.strip()
        st.session_state.chat_history.append({"role":"user","text":q})
        answer = _answer(q, d)
        if st.session_state.detail_level == "analyst":
            bm = d["bm"]; pm = d["primary_metric"]
            roc = bm.get("test_roc_auc") or bm.get(f"test_{pm}",0)
            answer += (f"\n\n📊 **Additional metrics:** {pm.upper()}={roc:.3f}  "
                       f"F1={bm.get('test_f1',0):.3f}  Acc={bm.get('test_accuracy',0):.3f}")
        elif st.session_state.detail_level == "technical":
            answer += f"\n\n🔬 For full technical details, switch to the **Technical Dashboard**."
        st.session_state.chat_history.append({"role":"bot","text":answer})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW — SUBMIT JOB
# ─────────────────────────────────────────────────────────────────────────────
elif "Submit" in view:
    st.markdown(f"""
    <div style="padding-bottom:18px;border-bottom:1px solid {BORDER};margin-bottom:26px;">
      <div style="font-size:21px;font-weight:800;color:{NAVY};letter-spacing:-.4px;">Submit New Analysis Job</div>
      <div style="font-size:13px;color:{SUB};margin-top:4px;">Upload your dataset and describe your business goal — AutoDS handles the full pipeline automatically</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 1 · Upload Dataset")
    uploaded = st.file_uploader("Select CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded:
        preview_df = pd.read_csv(uploaded)
        st.markdown(f'<div style="font-size:12px;color:{SUB};margin:8px 0 4px;">'
                    f'{uploaded.name} · {len(preview_df):,} rows · {len(preview_df.columns)} columns</div>',
                    unsafe_allow_html=True)
        st.dataframe(preview_df.head(6), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 2 · Describe Business Goal")
    business_desc = st.text_area(
        "Business Description",
        placeholder="e.g. We want to predict whether a customer will churn. The target column is churn, we want high recall to minimise false negatives.",
        height=110, label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 3 · Target Column & Task Type")
    ca, cb = st.columns(2)
    with ca:
        if uploaded:
            opts = ["(Let AutoDS infer automatically)"] + preview_df.columns.tolist()
            sel = st.selectbox("Target Column", opts)
            target_col = "" if "infer" in sel else sel
        else:
            target_col = st.text_input("Target Column (leave blank to let AutoDS infer)", placeholder="e.g. target")
    with cb:
        prob_sel = st.selectbox("Task Type", ["Auto-detect", "Classification", "Regression"])
        prob_map = {"Auto-detect":"classification","Classification":"classification","Regression":"regression"}
        problem_type = prob_map[prob_sel]

    st.markdown("<br>", unsafe_allow_html=True)
    can_run = uploaded is not None
    if not can_run:
        st.markdown(f'<div style="font-size:12px;color:{SUB};">Please upload a CSV file first.</div>',
                    unsafe_allow_html=True)

    run_btn = st.button("▶  Run AutoDS Pipeline", disabled=not can_run)

    if run_btn and can_run:
        tmp = PROJECT_ROOT / "_uploaded_data_temp.csv"
        tmp.write_bytes(uploaded.getvalue())
        with st.spinner("🔄 Running analysis pipeline, please wait…"):
            ok, err = run_pipeline(str(tmp), business_desc, target_col.strip(), problem_type)
        if ok:
            # Reset chat for the new run
            st.session_state.chat_history = []
            get_pipeline_data.clear()
            st.success("✅ Analysis complete! Switch to **Chat Analysis** to start asking questions.")
            st.balloons()
        else:
            st.error("Pipeline execution failed. Error details:")
            st.code(err, language="python")


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW — TECHNICAL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif "Dashboard" in view:

    results_exist = (BASE / "06_reports" / "pipeline_report_input.json").exists()
    if not results_exist:
        st.markdown(f"""<div style="text-align:center;padding:80px 0;">
          <div style="font-size:40px;margin-bottom:16px;">📭</div>
          <div style="font-size:18px;font-weight:700;color:{NAVY};margin-bottom:8px;">No Analysis Results</div>
          <div style="font-size:14px;color:{SUB};">Please submit a job to run the pipeline.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    d = get_pipeline_data()
    bm = d["bm"]; pm = d["primary_metric"]; pt = d["problem_type"]
    bmn = d["best_model_name"]; lb = d["lb_list"]
    roc  = bm.get("test_roc_auc") or bm.get(f"test_{pm}",0)
    f1   = bm.get("test_f1",0)
    acc  = bm.get("test_accuracy",0)
    prec = bm.get("test_precision",0)
    rec  = bm.get("test_recall",0)
    feat_imp = d["feat_imp"]

    # Sub-section navigation
    st.markdown("---")
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;'
                f'color:{SUB};padding:2px 0 8px;">Pipeline Stages</div>', unsafe_allow_html=True)

    with st.sidebar:
        dash_nav = st.radio("DashNav",
            ["⬡ Overview","◎ Data Understanding","◈ Data Cleaning","◆ Feature Engineering","◉ Modelling","◐ Evaluation","📄 Report"],
            label_visibility="collapsed")

    # ──────── Total overview ──────────────────────────────────────────────────
    if "Overview" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:18px;border-bottom:1px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:21px;font-weight:800;color:{NAVY};">Pipeline Overview</div>
          <div style="font-size:12px;color:{SUB};margin-top:3px;">Run time: {d["timestamp"]}</div>
        </div>""", unsafe_allow_html=True)

        steps  = ["Data Understanding","Data Cleaning","Feature Engineering","Modelling","Evaluation","Report"]
        subs   = ["Schema + Quality","Dedup · Impute","Select · Scale",
                  f"{len(lb)} candidates","Leaderboard","MD + JSON"]
        cs = st.columns(len(steps))
        for col, step, sub in zip(cs, steps, subs):
            col.markdown(f"""
            <div style="background:#fff;border:1px solid {BORDER};border-radius:10px;
                        padding:14px 10px;text-align:center;box-shadow:0 1px 4px rgba(14,54,124,.07);">
              <div style="width:22px;height:22px;border-radius:50%;background:#E8F7EE;
                          border:1.5px solid {GREEN};display:flex;align-items:center;
                          justify-content:center;margin:0 auto 8px;font-size:11px;
                          color:{GREEN};font-weight:700;">✓</div>
              <div style="font-size:11px;font-weight:600;color:#24364B;">{step}</div>
              <div style="font-size:10px;color:{SUB};margin-top:2px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        clean_sum = d["clean_sum"]
        k1.metric("Original Samples", d["n_rows_in"],  f"{d['in_shape'].get('columns','—')} raw features")
        k2.metric("Data Retention",
                  f"{clean_sum.get('data_retention_pct', clean_sum.get('data_retention_percentage', 0)):.0f}%",
                  f"Removed {clean_sum.get('rows_removed',0)} rows")
        k3.metric("Final Features", d["final_feat_cnt"], "after feature selection")
        k4.metric("Candidate Models", len(lb), f"Best: {bmn.replace('_',' ').title()}")
        k5.metric(f"Best {pm.upper()}", f"{roc:.3f}", "test set")

    # ──────── Data understanding ──────────────────────────────────────────────
    elif "Data Understanding" in dash_nav:
        st.markdown("## 01 · Data Understanding")
        dh = d["understanding"].get("downstream_handoff",{})
        fe = dh.get("feature_engineering_agent",{})
        mo = dh.get("modelling_agent",{})

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{SUB};text-transform:uppercase;'
                        f'letter-spacing:.06em;margin-bottom:8px;">Dataset Scale</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Metric":["Rows","Columns","Numeric Cols","Categorical Cols"],
                "Value":[d["n_rows_in"], d["in_shape"].get("columns","—"),
                      len(fe.get("numeric_columns",[])), len(fe.get("categorical_columns",[]))]
            }), hide_index=True, use_container_width=True)
        with c2:
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{SUB};text-transform:uppercase;'
                        f'letter-spacing:.06em;margin-bottom:8px;">Target Column</div>', unsafe_allow_html=True)
            imb_ratio = "—"
            for f in d["major_findings"]:
                m = re.search(r'\d+\.\d+', f)
                if m and ("imbalanc" in f.lower() or "ratio" in f.lower()):
                    imb_ratio = m.group(); break
            st.dataframe(pd.DataFrame({
                "Item":["Target Column","Task Type","Imbalance Ratio","Primary Metric"],
                "Value":[mo.get("target_column","—"), mo.get("problem_type","—").title(),
                      imb_ratio, pm.upper()]
            }), hide_index=True, use_container_width=True)
        with c3:
            st.markdown(f'<div style="font-size:12px;font-weight:600;color:{SUB};text-transform:uppercase;'
                        f'letter-spacing:.06em;margin-bottom:8px;">Quality Checks</div>', unsafe_allow_html=True)
            _dqm2 = d["cleaning"].get("data_quality_metrics", {})
            dqm = _dqm2.get("before_cleaning", _dqm2.get("original", {}))
            ids = fe.get("suspected_identifier_columns",[])
            _nulls = dqm.get("null_count", dqm.get("null_values", 0))
            _dups  = dqm.get("duplicate_rows", dqm.get("duplicates", 0))
            st.dataframe(pd.DataFrame({
                "Check":["Missing Values","Duplicate Rows","Constant Cols","ID-like Cols"],
                "Status":["✓ None" if _nulls == 0 else f"⚠ {_nulls}",
                         f"✓ {_dups}", "✓ 0",
                         f"⚠ {len(ids)}" if ids else "✓ None"]
            }), hide_index=True, use_container_width=True)

        st.markdown("<br>")
        st.markdown("### Key Findings")
        colors = [BLUE, ORANGE, GREEN, NAVY, CYAN]
        for i, f in enumerate(d["major_findings"]):
            c = ORANGE if "identifier" in f.lower() else (ORANGE if "imbalance" in f.lower() else BLUE)
            st.markdown(f"""<div class="det-row">
              <div class="det-dot" style="background:{c};"></div>
              <div style="font-size:13px;color:#24364B;">{f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>")
        st.markdown("### Risks & Next Steps")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Primary Risks**")
            for r in d["primary_risks"]:
                st.markdown(f'<div style="display:flex;gap:8px;padding:7px 0;font-size:13px;'
                            f'border-bottom:1px solid {BORDER};">'
                            f'<span style="color:{ORANGE};">⚠</span>{r}</div>', unsafe_allow_html=True)
        with rc2:
            st.markdown("**Recommended Steps**")
            for i, s in enumerate(d["next_steps"]):
                st.markdown(f'<div style="display:flex;gap:8px;padding:7px 0;font-size:13px;'
                            f'border-bottom:1px solid {BORDER};">'
                            f'<span style="color:{BLUE};font-weight:700;">{i+1}</span>{s}</div>',
                            unsafe_allow_html=True)

    # ──────── Cleaning ────────────────────────────────────────────────────────
    elif "Data Cleaning" in dash_nav:
        st.markdown("## 02 · Data Cleaning")
        cs = d["clean_sum"]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Input Rows",  d["n_rows_in"],  f"{d['in_shape'].get('columns','—')} cols")
        k2.metric("Output Rows", d["n_rows_out"], f"{d['out_shape'].get('columns','—')} cols")
        k3.metric("Removed Rows", cs.get("rows_removed",0), "Anomaly/Duplicate")
        _ret = float(cs.get('data_retention_pct', cs.get('data_retention_percentage', 0)) or 0)
        k4.metric("Retention", f"{_ret:.1f}%",
                  "✓ Excellent" if _ret >= 95 else "⚠ Low")
        st.markdown("<br>")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### Before vs After Cleaning")
            _dqm = d["cleaning"].get("data_quality_metrics", {})
            dqo = _dqm.get("before_cleaning", _dqm.get("original", {}))
            dqp = _dqm.get("after_cleaning",  _dqm.get("processed", {})) or {}
            st.dataframe(pd.DataFrame({
                "Metric":["Completeness (%)","Duplicate Rows","Null Values","Anomalous Rows"],
                "Before":[dqo.get("completeness_pct", dqo.get("completeness", 100)),
                           dqo.get("duplicate_rows",  dqo.get("duplicates", 0)),
                           dqo.get("null_count",       dqo.get("null_values", 0)),
                           cs.get("rows_removed", 0)],
                "After":[dqp.get("completeness_pct", dqp.get("completeness", 100)),
                           dqp.get("duplicate_rows",  dqp.get("duplicates", 0)),
                           dqp.get("null_count",       dqp.get("null_values", 0)),
                           0],
            }), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("### Operations Performed")
            ops = [("Column Name Normalisation","snake_case"),
                   (f"Remove Duplicate Rows",f"{dqo.get('duplicate_rows', dqo.get('duplicates', 0))} rows"),
                   ("Impute Missing Values","Median imputation"),
                   (f"IQR Outlier Filtering",f"Removed {cs.get('rows_removed',0)} rows")]
            for label, tag in ops:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:9px 12px;
                            background:#F4F6F9;border-radius:6px;margin-bottom:7px;font-size:13px;">
                  <span style="color:{GREEN};font-weight:700;">✓</span>
                  <span style="flex:1;">{label}</span>
                  <span style="font-size:11px;background:#EAF3FB;color:{BLUE};
                                border-radius:20px;padding:2px 9px;font-weight:600;">{tag}</span>
                </div>""", unsafe_allow_html=True)

    # ──────── Feature engineering ─────────────────────────────────────────────
    elif "Feature" in dash_nav:
        st.markdown("## 03 · Feature Engineering")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Final Features",  d["final_feat_cnt"],"after feature selection")
        k2.metric("Train Samples",   d["n_train"] or "—","~80%")
        k3.metric("Test Samples",    d["n_test"]  or "—","~20%")
        k4.metric("CV Folds", 5, "Stratified K-Fold")

        st.markdown("<br>")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### Feature Importance Ranking")
            fi = feat_imp
            if not fi.empty:
                mx = fi["importance"].max()
                for i, row in fi.iterrows():
                    p = row["importance"]/mx*100 if mx else 0
                    st.markdown(f"""
                    <div style="padding:11px 14px;background:#F4F6F9;border-radius:6px;
                                border:1px solid {BORDER};margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-weight:600;font-size:13px;color:#0E2F63;">{row['feature_name']}</span>
                        <span style="font-size:10px;color:{SUB};">#{i+1} · {int(row['importance'])}</span>
                      </div>
                      <div style="height:4px;background:{BORDER};border-radius:2px;overflow:hidden;">
                        <div style="height:100%;width:{p:.0f}%;background:linear-gradient(90deg,{BRIGHT},{CYAN});
                                    border-radius:2px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No feature importance data available.")
        with c2:
            st.markdown("### Train/Test Split")
            nt, nv = d["n_train"], d["n_test"]
            if nt and nv:
                fig = go.Figure(go.Pie(
                    labels=[f"Train ({nt})",f"Test ({nv})"],
                    values=[nt,nv], hole=0.68,
                    marker=dict(colors=[BLUE,CYAN],line=dict(color="#fff",width=3)),
                    hovertemplate="%{label}: %{value} samples<extra></extra>",
                ))
                fig.update_layout(showlegend=True,
                    legend=dict(orientation="h",y=-0.08,font=dict(size=12,color=SUB)),
                    margin=dict(t=10,b=20,l=0,r=0),height=240,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

    # ──────── Modelling ────────────────────────────────────────────────────────
    elif "Modelling" in dash_nav:
        st.markdown("## 04 · Modelling")
        lbl, _ = score_label(roc)
        st.markdown(f"""
        <div style="background:{NAVY};border-radius:12px;padding:24px 28px;margin-bottom:22px;color:#fff;">
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
            <div style="width:44px;height:44px;background:rgba(255,255,255,.12);border-radius:11px;
                        display:flex;align-items:center;justify-content:center;font-size:22px;">🏆</div>
            <div>
              <div style="font-size:20px;font-weight:800;">{bmn.replace("_"," ").title()}</div>
              <div style="font-size:12px;color:rgba(255,255,255,.55);margin-top:2px;">
                Best Model · Rank #1 by {pm.upper()} · {len(lb)} candidates total
              </div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:34px;font-weight:800;color:#7DDFF5;">{roc:.3f}</div>
              <div style="font-size:11px;color:rgba(255,255,255,.5);">Test Set {pm.upper()}</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            {''.join(f"""<div style="background:rgba(255,255,255,.08);border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:20px;font-weight:800;">{v:.3f}</div>
              <div style="font-size:10px;color:rgba(255,255,255,.5);margin-top:3px;">{lbl2}</div>
            </div>""" for v,lbl2 in [(acc,"Accuracy"),(prec,"Precision"),(rec,"Recall"),(f1,"F1")])}
          </div>
        </div>""", unsafe_allow_html=True)

        mc1,mc2 = st.columns(2)
        with mc1:
            st.markdown("### Test Set Metrics")
            for label, val in [("ROC-AUC",roc),("F1",f1),("Recall",rec),("Precision",prec),("Accuracy",acc)]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                  <div style="width:82px;font-size:12px;color:{SUB};flex-shrink:0;">{label}</div>
                  <div style="flex:1;height:6px;background:#EAF3FB;border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{val*100:.1f}%;background:linear-gradient(90deg,{BLUE},{BRIGHT});
                                border-radius:3px;"></div>
                  </div>
                  <div style="width:40px;text-align:right;font-size:12px;font-weight:700;color:#0E2F63;">{val:.3f}</div>
                </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown("### Radar Chart — Top 2 Models")
            if len(lb) >= 2:
                cats=["ROC-AUC","F1","Recall","Precision","Accuracy"]
                fig=go.Figure()
                for m,color,fill in zip(lb[:2],[BLUE,CYAN],
                                        ["rgba(35,97,174,.15)","rgba(59,165,199,.1)"]):
                    vals=[m["test_roc_auc"]*100,m["test_f1"]*100,m["test_recall"]*100,
                          m["test_precision"]*100,m["test_accuracy"]*100]
                    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
                        fill="toself",fillcolor=fill,line=dict(color=color,width=2),
                        name=m["model_name"].replace("_"," ").title(),
                        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>"))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True,range=[30,100],showticklabels=False,
                               gridcolor=BORDER,linecolor=BORDER),
                               angularaxis=dict(gridcolor=BORDER,linecolor=BORDER),
                               bgcolor="rgba(0,0,0,0)"),
                    showlegend=True,legend=dict(orientation="h",y=-0.12,font=dict(size=11,color=SUB)),
                    margin=dict(t=10,b=40,l=20,r=20),height=260,
                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        if not feat_imp.empty:
            st.markdown("### Feature Importance")
            fig=go.Figure(go.Bar(
                x=feat_imp["importance"],y=feat_imp["feature_name"],orientation="h",
                marker=dict(color=[NAVY,BLUE,CYAN][:len(feat_imp)],cornerradius=4),
                hovertemplate="%{y}: %{x}<extra></extra>"))
            fig.update_layout(
                xaxis=dict(gridcolor=BORDER,tickfont=dict(color=SUB,size=12)),
                yaxis=dict(showgrid=False,tickfont=dict(color="#24364B",size=12)),
                margin=dict(t=10,b=10,l=0,r=0),height=max(120,len(feat_imp)*45),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Model Leaderboard")
        if not d["leaderboard"].empty:
            lb_df = d["leaderboard"].copy()
            show_cols = [c for c in ["rank","model_name","test_roc_auc","test_f1",
                                     "test_accuracy","test_precision","test_recall",
                                     "cv_roc_auc","cv_runtime_seconds"] if c in lb_df.columns]
            display = lb_df[show_cols].copy()
            display.columns = [c.replace("_"," ").title() for c in show_cols]
            for col in display.columns:
                if display[col].dtype == float:
                    display[col] = display[col].round(4)
            st.dataframe(display, hide_index=True, use_container_width=True)

    # ──────── Evaluation ──────────────────────────────────────────────────────
    elif "Evaluation" in dash_nav:
        st.markdown("## 05 · Evaluation")
        ec1,ec2 = st.columns(2)
        with ec1:
            st.markdown("### ROC-AUC Comparison")
            if lb:
                names=[m["model_name"].replace("_"," ").title() for m in lb]
                palette=[NAVY,BLUE,BRIGHT,CYAN,MIST]
                fig=go.Figure()
                fig.add_trace(go.Bar(name="Test Set",x=names,
                    y=[m["test_roc_auc"] for m in lb],
                    marker=dict(color=palette[:len(names)],cornerradius=5),
                    hovertemplate="%{x}: %{y:.3f}<extra>Test</extra>"))
                fig.add_trace(go.Bar(name="CV Mean",x=names,
                    y=[m["cv_roc_auc"] for m in lb],
                    marker=dict(color="rgba(35,97,174,.12)",
                                line=dict(color="rgba(35,97,174,.3)",width=1),cornerradius=5),
                    hovertemplate="%{x}: %{y:.3f}<extra>CV</extra>"))
                fig.update_layout(barmode="group",
                    yaxis=dict(range=[0,1],gridcolor=BORDER,tickfont=dict(color=SUB)),
                    xaxis=dict(showgrid=False,tickfont=dict(color="#24364B",size=11)),
                    legend=dict(orientation="h",y=-0.18,font=dict(size=11,color=SUB)),
                    margin=dict(t=10,b=50,l=0,r=0),height=260,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with ec2:
            st.markdown("### Multi-Metric Comparison")
            if lb:
                names=[m["model_name"].replace("_"," ").title() for m in lb]
                fig=go.Figure()
                for mname,key,mc in [("Accuracy","test_accuracy","rgba(14,54,124,.8)"),
                                      ("Precision","test_precision","rgba(35,97,174,.8)"),
                                      ("Recall","test_recall","rgba(43,124,197,.8)"),
                                      ("F1","test_f1","rgba(59,165,199,.8)")]:
                    fig.add_trace(go.Bar(name=mname,x=names,y=[m[key] for m in lb],
                        marker=dict(color=mc,cornerradius=3),
                        hovertemplate=f"{mname} — %{{x}}: %{{y:.3f}}<extra></extra>"))
                fig.update_layout(barmode="group",
                    yaxis=dict(range=[0,1],gridcolor=BORDER,tickfont=dict(color=SUB)),
                    xaxis=dict(showgrid=False,tickfont=dict(color="#24364B",size=10)),
                    legend=dict(orientation="h",y=-0.18,font=dict(size=11,color=SUB)),
                    margin=dict(t=10,b=50,l=0,r=0),height=260,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Best Model Selection Rationale")
        se1,se2,se3 = st.columns(3)
        se1.metric("Selected Model",   bmn.replace("_"," ").title())
        se2.metric("Selection Metric", pm.upper())
        se3.metric("Test Score",       f"{roc:.3f}")

        st.markdown("### Deployment Recommendations")
        lbl, _ = score_label(roc)
        recs = [
            ("green","✅",f"Deploy {bmn.replace('_',' ').title()} to staging environment",
             "Model is stable on both test set and cross-validation."),
            ("warn","⚠","Monitor Data Distribution Drift","Retrain periodically as new data accumulates."),
            ("blue","↗","Collect More Labelled Data",f"Current {pm.upper()}={roc:.3f}, there is still room for improvement."),
        ]
        ids = d["understanding"].get("downstream_handoff",{}).get(
            "feature_engineering_agent",{}).get("suspected_identifier_columns",[])
        if ids:
            recs.append(("navy","◎","Check ID-like Column Leakage Risk",
                         f"{', '.join(ids)} appears to be identifier columns — confirm no data leakage before deployment."))
        alert_styles={"green":(GREEN,"#E8F7EE","#A8D5B5"),"warn":(ORANGE,"#FEF3E2","#F8D89C"),
                      "blue":(BLUE,"#EAF3FB","#B8D8F0"),"navy":(NAVY,"#E8EEF6","#C2CDE0")}
        for key,icon,title,desc in recs:
            _,bg,border=alert_styles[key]
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:12px 14px;border-radius:6px;
                        background:{bg};border:1px solid {border};margin-bottom:8px;align-items:flex-start;">
              <div style="font-size:15px;">{icon}</div>
              <div>
                <div style="font-weight:600;font-size:13px;color:#0E2F63;">{title}</div>
                <div style="font-size:12px;color:{SUB};margin-top:2px;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ──────── Report ──────────────────────────────────────────────────────────
    elif "Report" in dash_nav:
        st.markdown("## 06 · Report")
        rt1, rt2 = st.tabs(["📋 Business Report", "🔬 Technical Report"])
        with rt1:
            br = d.get("business_report","")
            if br:
                st.markdown(br)
            else:
                st.info("No business report available.")
        with rt2:
            tr = d["report"].get("technical_report", d["report"].get("technical", ""))
            if tr:
                st.markdown(tr)
            else:
                st.info("No technical report available.")
