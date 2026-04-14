"""
AutoDS  ·  Modern Dashboard  (redesigned)
Same backend / same brand colors — more modern aesthetic
"""

from __future__ import annotations
import json, os, re, sys, traceback
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE = PROJECT_ROOT / "autods_pipeline_output"

# ── Brand tokens (unchanged) ──────────────────────────────────────────────────
NAVY   = "#0E367C"
BLUE   = "#2361AE"
BRIGHT = "#2B7CC5"
CYAN   = "#3BA5C7"
MIST   = "#9CB7C8"
GREEN  = "#1B8A56"
ORANGE = "#D97706"
BG     = "#F4F7FC"
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
# GLOBAL CSS  — modern / glass-card aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Noto+Sans+SC:wght@400;500;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
  font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  background: {BG};
  color: #1E2D3D;
}}
.main .block-container {{
  padding: 2rem 2.5rem 3rem;
  max-width: 1280px;
}}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] > div {{
  background: linear-gradient(175deg, {NAVY} 0%, #0c2d6b 60%, #071e4a 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}}
section[data-testid="stSidebar"] * {{ color: rgba(255,255,255,.88) !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.10) !important; }}
section[data-testid="stSidebar"] .stRadio label {{
  font-size: 13px !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  transition: background .15s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
  background: rgba(255,255,255,.08) !important;
}}

/* ── Streamlit metric override ─────────────────────────────────────────── */
[data-testid="metric-container"] {{
  background: {WHITE};
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 20px 22px 16px;
  box-shadow: 0 2px 12px rgba(14,54,124,.07);
  transition: box-shadow .2s;
}}
[data-testid="metric-container"]:hover {{
  box-shadow: 0 4px 20px rgba(14,54,124,.12);
}}
[data-testid="metric-container"] label {{
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: .09em;
  color: {SUB} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
  font-size: 26px !important;
  font-weight: 800 !important;
  color: {NAVY} !important;
  letter-spacing: -.5px;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
  font-size: 11px !important;
  font-weight: 600 !important;
}}

/* ── Headings ──────────────────────────────────────────────────────────── */
h1, h2, h3 {{ font-family: 'Inter', 'Noto Sans SC', sans-serif !important; }}
h2 {{ color: {NAVY} !important; font-weight: 800 !important; letter-spacing: -.4px; }}
h3 {{ color: #0E2F63 !important; font-weight: 700 !important; font-size: 14px !important;
      letter-spacing: -.2px; text-transform: uppercase; opacity: .7; margin-top: 0 !important; }}
hr {{ border-color: {BORDER} !important; margin: 6px 0 18px; }}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {{
  background: {NAVY};
  color: #fff;
  border: none;
  border-radius: 10px;
  font-weight: 700;
  font-size: 13px;
  padding: 10px 24px;
  letter-spacing: .02em;
  transition: background .18s, transform .12s, box-shadow .18s;
  box-shadow: 0 2px 8px rgba(14,54,124,.20);
}}
.stButton > button:hover {{
  background: {BLUE};
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(14,54,124,.28);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* Danger / secondary button */
.stButton > button[kind="secondary"] {{
  background: transparent;
  color: {SUB} !important;
  border: 1px solid {BORDER} !important;
  box-shadow: none;
}}
.stButton > button[kind="secondary"]:hover {{
  border-color: {BLUE} !important;
  color: {BLUE} !important;
  transform: none;
  box-shadow: none;
}}

/* ── Dataframe styling ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
  border: 1px solid {BORDER};
  border-radius: 10px;
  overflow: hidden;
}}

/* ── Form inputs ───────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div {{
  border-radius: 9px !important;
  border: 1px solid {BORDER} !important;
  font-size: 13px !important;
  transition: border-color .18s, box-shadow .18s;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: {BLUE} !important;
  box-shadow: 0 0 0 3px rgba(35,97,174,.12) !important;
}}

/* ── Chat bubbles ──────────────────────────────────────────────────────── */
.msg-user {{
  background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
  color: #fff;
  border-radius: 20px 20px 5px 20px;
  padding: 13px 18px;
  margin: 0 0 6px auto;
  max-width: 72%;
  font-size: 14px;
  line-height: 1.6;
  width: fit-content;
  box-shadow: 0 3px 12px rgba(14,54,124,.22);
}}
.msg-bot {{
  background: {WHITE};
  color: #1E2D3D;
  border-radius: 20px 20px 20px 5px;
  border: 1px solid {BORDER};
  padding: 15px 20px;
  margin: 0 auto 6px 0;
  max-width: 86%;
  font-size: 13.5px;
  line-height: 1.7;
  width: fit-content;
  box-shadow: 0 2px 8px rgba(14,54,124,.06);
}}
.msg-bot b {{ color: {NAVY}; }}
.msg-bot code {{
  background: #EAF3FB;
  color: {BLUE};
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
  font-family: 'SF Mono', 'Fira Code', monospace;
}}

/* ── Glass card ────────────────────────────────────────────────────────── */
.glass-card {{
  background: rgba(255,255,255,.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(221,228,236,.8);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 24px rgba(14,54,124,.08), 0 1px 4px rgba(14,54,124,.04);
  transition: box-shadow .2s;
}}
.glass-card:hover {{ box-shadow: 0 8px 32px rgba(14,54,124,.12), 0 2px 8px rgba(14,54,124,.06); }}

/* ── Gradient hero banner ──────────────────────────────────────────────── */
.hero-banner {{
  background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 55%, {BRIGHT} 100%);
  border-radius: 16px;
  padding: 28px 32px;
  color: #fff;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}}
.hero-banner::before {{
  content: '';
  position: absolute;
  top: -40%;
  right: -5%;
  width: 360px;
  height: 360px;
  background: rgba(255,255,255,.04);
  border-radius: 50%;
  pointer-events: none;
}}
.hero-banner::after {{
  content: '';
  position: absolute;
  bottom: -60%;
  right: 15%;
  width: 240px;
  height: 240px;
  background: rgba(59,165,199,.12);
  border-radius: 50%;
  pointer-events: none;
}}

/* ── Stage status pill ─────────────────────────────────────────────────── */
.stage-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(27,138,86,.12);
  border: 1px solid rgba(27,138,86,.3);
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 700;
  color: {GREEN};
  letter-spacing: .04em;
}}

/* ── Stat badge ─────────────────────────────────────────────────────────── */
.stat-badge {{
  display: inline-block;
  background: #EAF3FB;
  color: {BLUE};
  border-radius: 6px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .03em;
}}

/* ── Section header ────────────────────────────────────────────────────── */
.section-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid {BORDER};
}}
.section-icon {{
  width: 34px;
  height: 34px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}}

/* ── Finding row ────────────────────────────────────────────────────────── */
.finding-row {{
  display: flex;
  gap: 12px;
  padding: 11px 14px;
  background: #fff;
  border: 1px solid {BORDER};
  border-radius: 10px;
  margin-bottom: 8px;
  align-items: flex-start;
  transition: box-shadow .15s;
}}
.finding-row:hover {{
  box-shadow: 0 2px 10px rgba(14,54,124,.08);
}}
.finding-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 5px;
}}

/* ── Op row (cleaning operations) ──────────────────────────────────────── */
.op-row {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: #F8FAFD;
  border: 1px solid {BORDER};
  border-radius: 10px;
  margin-bottom: 8px;
  font-size: 13px;
}}

/* ── Metric bar ─────────────────────────────────────────────────────────── */
.mbar-wrap {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}}
.mbar-label {{ width: 80px; font-size: 12px; color: {SUB}; flex-shrink: 0; }}
.mbar-track {{
  flex: 1;
  height: 7px;
  background: #EAF3FB;
  border-radius: 4px;
  overflow: hidden;
}}
.mbar-fill {{
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, {BLUE}, {CYAN});
  transition: width .4s cubic-bezier(.4,0,.2,1);
}}
.mbar-val {{ width: 44px; text-align: right; font-size: 12px; font-weight: 700; color: {NAVY}; }}

/* ── Alert row ──────────────────────────────────────────────────────────── */
.alert-row {{
  display: flex;
  gap: 14px;
  padding: 14px 16px;
  border-radius: 12px;
  margin-bottom: 10px;
  align-items: flex-start;
}}
.alert-icon {{ font-size: 17px; flex-shrink: 0; margin-top: 1px; }}
.alert-title {{ font-weight: 700; font-size: 13px; }}
.alert-desc  {{ font-size: 12px; margin-top: 3px; line-height: 1.5; }}

/* ── Navigation radio override ─────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
  gap: 4px !important;
}}

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {{ color: {BLUE} !important; }}

/* ── Hide streamlit branding ────────────────────────────────────────────── */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ── Feature bar ─────────────────────────────────────────────────────────── */
.feat-bar {{
  padding: 13px 16px;
  background: #F8FAFD;
  border: 1px solid {BORDER};
  border-radius: 12px;
  margin-bottom: 8px;
  transition: box-shadow .15s;
}}
.feat-bar:hover {{ box-shadow: 0 2px 10px rgba(14,54,124,.08); }}
.feat-bar-top {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}}
.feat-name {{ font-weight: 700; font-size: 13px; color: #0E2F63; }}
.feat-rank {{ font-size: 10px; color: {SUB}; }}
.feat-track {{
  height: 5px;
  background: {BORDER};
  border-radius: 3px;
  overflow: hidden;
}}
.feat-fill {{
  height: 100%;
  border-radius: 3px;
  background: linear-gradient(90deg, {BRIGHT}, {CYAN});
}}

/* ── Pipeline step cards ─────────────────────────────────────────────────── */
.step-card {{
  background: #fff;
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 18px 14px;
  text-align: center;
  box-shadow: 0 2px 10px rgba(14,54,124,.06);
  transition: box-shadow .2s, transform .2s;
  height: 100%;
}}
.step-card:hover {{
  box-shadow: 0 6px 20px rgba(14,54,124,.12);
  transform: translateY(-2px);
}}
.step-check {{
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, {GREEN} 0%, #22c55e 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 10px;
  font-size: 13px;
  color: #fff;
  font-weight: 800;
  box-shadow: 0 2px 8px rgba(27,138,86,.3);
}}
.step-name {{ font-size: 12px; font-weight: 700; color: #24364B; margin-bottom: 4px; }}
.step-sub  {{ font-size: 10px; color: {SUB}; }}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 4px;
  border-bottom: 2px solid {BORDER};
  padding-bottom: 0;
}}
[data-testid="stTabs"] [role="tab"] {{
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
  border-radius: 8px 8px 0 0 !important;
  color: {SUB} !important;
  border: none !important;
  background: transparent !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  color: {NAVY} !important;
  background: rgba(35,97,174,.06) !important;
  border-bottom: 2px solid {BLUE} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (identical logic to streamlit_app.py)
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
    if v >= 0.95: return "Excellent", GREEN
    if v >= 0.85: return "Good",      BLUE
    if v >= 0.70: return "Acceptable", ORANGE
    return "Needs Work", "#C0392B"

def friendly_metric(metric_key: str, value: float, problem_type: str) -> str:
    label, _ = score_label(value)
    if problem_type == "regression":
        msgs = {
            "rmse": f"预测误差（RMSE）为 {value:.3f}，表示模型预测值与真实值的平均偏差。",
            "mae":  f"平均绝对误差（MAE）为 {value:.3f}。",
            "r2":   f"模型解释了 {value*100:.1f}% 的目标变量方差（R²={value:.3f}）。",
        }
        return msgs.get(metric_key, f"{metric_key.upper()} = {value:.3f}")
    else:
        msgs = {
            "roc_auc":   f"模型 ROC-AUC={value:.3f}（{label}），能较好地区分不同类别。",
            "f1":        f"F1={value:.3f}（{label}），综合了查全率和查准率。",
            "accuracy":  f"准确率 {value*100:.1f}%（{label}）。",
            "precision": f"查准率 {value*100:.1f}%（{label}）——预测为正例时的可信度。",
            "recall":    f"查全率 {value*100:.1f}%（{label}）——实际正例被发现的比例。",
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

    mod_d           = report_input.get("modeling", {})
    best_model_name = mod_d.get("best_model_name") or eval_summary.get("best_model_name", "—")
    primary_metric  = mod_d.get("primary_metric", "roc_auc")
    lb_list         = mod_d.get("leaderboard", [])
    feat_eng        = report_input.get("feature_engineering", {})
    final_feat_cnt  = feat_eng.get("final_feature_count", len(feat_imp))

    clean_sum  = cleaning.get("cleaning_summary", {})
    in_shape   = cleaning.get("input_data",  {}).get("shape", {})
    out_shape  = cleaning.get("output_data", {}).get("shape", {})
    n_rows_in  = in_shape.get("rows", "—")
    n_rows_out = out_shape.get("rows", "—")

    bm = best_metrics if best_metrics else (lb_list[0] if lb_list else {})

    yt = load_csv("03_feature_engineering/y_train.csv")
    yv = load_csv("03_feature_engineering/y_test.csv")

    dh = understanding.get("downstream_handoff", {})
    mo = dh.get("modelling_agent", {})

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
        major_findings=understanding.get("major_findings", []),
        primary_risks=understanding.get("primary_risks", []),
        next_steps=understanding.get("recommended_next_steps", []),
        exec_summary=understanding.get("executive_summary", ""),
        problem_type=mo.get("problem_type", "classification"),
        target_column=mo.get("target_column", "—"),
        business_report=report.get("business", ""),
        timestamp=report_input.get("meta", {}).get("timestamp", "—"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT ENGINE  (identical logic)
# ─────────────────────────────────────────────────────────────────────────────
GREETINGS = {"你好", "hi", "hello", "嗨", "开始", "start", "help", "帮助", "hey"}

def _opening_msg(d: dict) -> str:
    pt  = d["problem_type"]
    tgt = d["target_column"]
    bm  = d["bm"]
    pm  = d["primary_metric"]
    roc = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
    lbl, _ = score_label(roc)
    risks = d["primary_risks"]
    risk_hint = f"数据存在 {len(risks)} 个潜在风险需要关注。" if risks else "数据质量整体良好。"
    pt_cn = "回归（预测连续数值）" if pt == "regression" else "分类（预测类别标签）"
    return (
        f"您好👋 我已完成对您数据集的全流程分析。\n\n"
        f"**任务类型：** {pt_cn}\n"
        f"**目标列：** `{tgt}`\n"
        f"**最优模型表现：** {lbl}（{pm.upper()} = {roc:.3f}）\n"
        f"**数据状态：** {risk_hint}\n\n"
        "您可以继续问我：\n"
        "• 这份数据有什么问题？\n"
        "• 模型结果准吗？\n"
        "• 哪些变量最重要？\n"
        "• 模型可以上线吗？\n"
        "• 下一步应该做什么？"
    )

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

    if any(w in q for w in GREETINGS):
        return _opening_msg(d)

    if any(w in q for w in ["数据", "dataset", "数据集", "data", "概况", "overview", "情况", "summary"]):
        es = d["exec_summary"]
        if es:
            findings = "\n".join(f"• {f}" for f in d["major_findings"][:4])
            return f"{es}\n\n**主要发现：**\n{findings}"
        return "暂无数据理解结果，请先运行流水线。"

    if any(w in q for w in ["风险", "risk", "问题", "issue", "danger", "警告", "warning"]):
        risks = d["primary_risks"]
        if not risks:
            return "未发现明显数据风险，数据质量整体良好。"
        lines = "\n".join(f"• {r}" for r in risks)
        return f"检测到以下潜在风险：\n\n{lines}\n\n建议在上线前逐一核查。"

    if any(w in q for w in ["下一步", "next", "建议", "recommend", "action", "应该", "怎么做", "what should"]):
        steps = d["next_steps"]
        if not steps:
            return "流水线已完成，建议将模型部署到测试环境并持续监控性能。"
        lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        br = d.get("business_report", "")
        rec_section = ""
        if "recommend" in br.lower() or "建议" in br or "action" in br.lower():
            for line in br.split("\n"):
                if any(k in line.lower() for k in ["recommend", "action", "建议", "next"]):
                    rec_section = line.strip(); break
        return f"根据分析结果，建议按以下顺序处理：\n\n{lines}" + (f"\n\n{rec_section}" if rec_section else "")

    if any(w in q for w in ["模型", "model", "结果", "result", "准吗", "准确", "performance", "表现", "效果", "怎么样"]):
        score_str = friendly_metric(pm, roc, pt)
        lbl, _ = score_label(roc)
        imb_flag = ""
        for f in d["major_findings"]:
            if "imbalance" in f.lower() or "不平衡" in f:
                imb_flag = "\n\n⚠ 注意：目标列存在类别不平衡，建议重点关注 F1 或 Recall 指标。"
                break
        return (
            f"**最优模型：{bmn.replace('_',' ').title()}**  ·  综合评级：**{lbl}**\n\n"
            f"{score_str}\nF1={fmt(f1)}  ·  Accuracy={pct(acc)}\n"
            f"{imb_flag}\n\n共比较了 {len(d['lb_list'])} 个候选模型，{bmn.replace('_',' ').title()} 综合表现最优。"
        )

    if any(w in q for w in ["部署", "deploy", "上线", "production", "生产", "能用吗", "可以用", "可用"]):
        lbl, _ = score_label(roc)
        if roc >= 0.85:
            verdict = "✅ 当前模型性能良好，可以考虑部署到测试环境进行进一步验证。"
        elif roc >= 0.70:
            verdict = "⚠ 当前模型性能中等，建议收集更多数据后再考虑生产部署。"
        else:
            verdict = "❌ 当前模型性能较弱，不建议直接上线，建议优化数据或模型后再评估。"
        return (
            f"{verdict}\n\n**部署前清单：**\n"
            "1. 让业务侧专家验证预测结果的业务逻辑\n"
            "2. 建立模型性能监控（数据分布漂移报警）\n"
            "3. 准备模型回退方案\n"
            "4. 目标列语义与实际业务对齐确认"
        )

    if any(w in q for w in ["特征", "feature", "变量", "variable", "重要", "important", "哪些", "which"]):
        fi = d["feat_imp"]
        if fi.empty:
            return "暂无特征重要性信息。"
        top3 = fi.head(3)
        lines = "\n".join(
            f"{i+1}. **{row['feature_name']}**（重要性评分：{int(row['importance'])}）"
            for i, row in top3.iterrows()
        )
        return (
            f"对预测结果影响最大的前三个特征为：\n\n{lines}\n\n"
            "这些变量是模型获取预测能力的主要来源，如果业务上这些变量难以获取，"
            "需要重新评估模型的可部署性。"
        )

    if any(w in q for w in ["为什么", "why", "判断", "infer", "分类", "regression", "回归", "classification", "任务", "task"]):
        if pt == "regression":
            return (
                f"因为目标列 **{tgt}** 是数值型，并且取值种类较多（连续变化），"
                "所以系统将其识别为**回归任务**（预测连续数值），而非分类任务。\n\n"
                "如果您认为它应该是分类任务，请在提交时明确指定问题类型。"
            )
        else:
            return (
                f"系统将目标列 **{tgt}** 识别为**分类任务**，"
                "因为其取值为离散类别（例如 是/否、多个类别标签）。\n\n"
                "如果您认为它应该是回归任务，请在提交时明确指定问题类型。"
            )

    if any(w in q for w in ["清洗", "clean", "缺失", "missing", "脏", "dirty", "处理", "preprocess"]):
        cs = d["clean_sum"]
        rows_removed = cs.get("rows_removed", 0)
        retention    = cs.get("data_retention_percentage", 0)
        n_in  = d["n_rows_in"]
        n_out = d["n_rows_out"]
        return (
            f"数据清洗已完成。\n\n"
            f"• 输入：{n_in} 行\n"
            f"• 输出：{n_out} 行\n"
            f"• 移除了 {rows_removed} 行异常/重复数据\n"
            f"• 数据保留率：{retention:.1f}%\n\n"
            f"{'✅ 数据质量良好，保留率高于 95%。' if float(retention or 0) >= 95 else '⚠ 数据保留率偏低，建议检查原始数据质量。'}"
        )

    if any(w in q for w in ["报告", "report", "总结", "conclusion", "业务", "business", "executive"]):
        br = d.get("business_report", "")
        if br:
            lines = [l.strip() for l in br.split("\n") if l.strip() and not l.startswith("#")]
            snippet = " ".join(lines[:6])[:600]
            return f'**业务报告摘要：**\n\n{snippet}\n\n（如需完整技术报告，请切换到"技术仪表板"视图）'
        return "暂无业务报告，请先运行流水线。"

    if any(w in q for w in ["排行", "leaderboard", "compare", "对比", "所有模型", "candidate", "候选"]):
        lb = d["lb_list"]
        if not lb:
            return "暂无模型排行榜数据。"
        lines = []
        for i, m in enumerate(lb[:5]):
            lines.append(
                f"{i+1}. **{m['model_name'].replace('_',' ').title()}** — "
                f"ROC-AUC {m.get('test_roc_auc',0):.3f}  F1 {m.get('test_f1',0):.3f}"
            )
        return "**模型排行榜（按 ROC-AUC 排序）：**\n\n" + "\n".join(lines)

    return (
        "我可以回答以下问题，请直接点击或输入：\n\n"
        "• 这份数据是什么情况？\n"
        "• 模型结果怎么样？\n"
        "• 有哪些风险？\n"
        "• 哪些变量最重要？\n"
        "• 下一步应该做什么？\n"
        "• 模型可以上线吗？"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detail_level" not in st.session_state:
    st.session_state.detail_level = "business"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
results_exist = (BASE / "06_reports" / "pipeline_report_input.json").exists()

with st.sidebar:
    # ── Brand mark ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 20px 4px 22px;">
      <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <div style="background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.15);
                    border-radius: 11px; width: 42px; height: 42px;
                    display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="4"  cy="18" r="2.5" fill="#2361AE"/>
            <circle cx="12" cy="10" r="2.5" fill="#2B7CC5"/>
            <circle cx="20" cy="5"  r="2.5" fill="#3BA5C7"/>
            <path d="M6.5 17 Q10 12 9.8 10"  stroke="#2361AE" stroke-width="1.7"
                  fill="none" stroke-linecap="round"/>
            <path d="M14.2 9.5 Q17 6.5 17.5 5" stroke="#3BA5C7" stroke-width="1.7"
                  fill="none" stroke-linecap="round"/>
            <path d="M20 7.5 Q21 13 17 16 Q13 19 9 18 Q6 17.5 5 20" stroke="#9CB7C8"
                  stroke-width="1.4" fill="none" stroke-linecap="round" stroke-dasharray="2 2.5"/>
          </svg>
        </div>
        <div>
          <div style="font-size: 18px; font-weight: 900; color: #fff; letter-spacing: -.3px;
                      line-height: 1;">AUTODS</div>
          <div style="font-size: 10px; color: rgba(255,255,255,.38); letter-spacing: .08em;
                      margin-top: 2px; text-transform: uppercase;">Auto Data Science</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.10);margin-bottom:16px;"></div>',
                unsafe_allow_html=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:9px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;'
        'color:rgba(255,255,255,.28);padding:0 2px 8px;">导航</div>',
        unsafe_allow_html=True,
    )
    view = st.radio(
        "View",
        ["💬  对话分析", "🚀  提交新任务", "📊  技术仪表板"],
        label_visibility="collapsed",
    )

    if results_exist:
        st.markdown('<div style="height:1px;background:rgba(255,255,255,.10);margin:16px 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:9px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;'
            'color:rgba(255,255,255,.28);padding:0 2px 8px;">回答深度</div>',
            unsafe_allow_html=True,
        )
        _level_opts = ["🟢  普通用户", "🟡  分析师", "🔵  技术细节"]
        _level_map  = {"🟢  普通用户": "business", "🟡  分析师": "analyst", "🔵  技术细节": "technical"}
        _rev_map    = {v: k for k, v in _level_map.items()}
        level = st.radio(
            "Level",
            _level_opts,
            label_visibility="collapsed",
            index=_level_opts.index(_rev_map.get(st.session_state.detail_level, "🟢  普通用户")),
        )
        st.session_state.detail_level = _level_map[level]

        st.markdown('<div style="height:1px;background:rgba(255,255,255,.10);margin:16px 0;"></div>',
                    unsafe_allow_html=True)
        if st.button("🗑  清空对话"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Status indicator ─────────────────────────────────────────────────────
    st.markdown('<div style="height:1px;background:rgba(255,255,255,.10);margin:16px 0 12px;"></div>',
                unsafe_allow_html=True)
    if results_exist:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:10px 12px;
                    background:rgba(27,138,86,.18);border:1px solid rgba(27,138,86,.3);
                    border-radius:10px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;
                      box-shadow:0 0 6px rgba(34,197,94,.6);flex-shrink:0;"></div>
          <div style="font-size:11px;font-weight:700;color:rgba(255,255,255,.9);">流水线结果可用</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:10px 12px;
                    background:rgba(217,119,6,.12);border:1px solid rgba(217,119,6,.25);
                    border-radius:10px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#f59e0b;flex-shrink:0;"></div>
          <div style="font-size:11px;font-weight:700;color:rgba(255,255,255,.7);">等待分析任务</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██  VIEW — CHAT
# ─────────────────────────────────────────────────────────────────────────────
if "对话" in view:

    if not results_exist:
        st.markdown(f"""
        <div style="text-align:center;padding:100px 0 80px;max-width:480px;margin:0 auto;">
          <div style="font-size:52px;margin-bottom:20px;">📭</div>
          <div style="font-size:22px;font-weight:800;color:{NAVY};margin-bottom:10px;
                      letter-spacing:-.4px;">还没有分析结果</div>
          <div style="font-size:14px;color:{SUB};line-height:1.6;">
            请先切换到 <strong style="color:{BLUE};">提交新任务</strong>，
            上传数据并运行流水线，完成后即可在此对话。
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    d = get_pipeline_data()

    # Auto-send opening message
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "bot", "text": _opening_msg(d)})

    # ── Hero header ──────────────────────────────────────────────────────────
    roc_hero = d["bm"].get("test_roc_auc") or d["bm"].get(f"test_{d['primary_metric']}", 0)
    lbl_hero, col_hero = score_label(roc_hero)
    st.markdown(f"""
    <div class="hero-banner">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <div style="font-size:22px;font-weight:900;letter-spacing:-.5px;margin-bottom:6px;">
            AI 数据顾问
          </div>
          <div style="font-size:13px;color:rgba(255,255,255,.65);max-width:400px;line-height:1.5;">
            基于您的数据分析结果，用自然语言回答您的业务问题
          </div>
        </div>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
          <div style="text-align:center;">
            <div style="font-size:11px;color:rgba(255,255,255,.45);letter-spacing:.06em;
                        text-transform:uppercase;margin-bottom:4px;">目标列</div>
            <div style="font-size:15px;font-weight:800;color:#7DDFF5;">{d["target_column"]}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:11px;color:rgba(255,255,255,.45);letter-spacing:.06em;
                        text-transform:uppercase;margin-bottom:4px;">最优模型</div>
            <div style="font-size:15px;font-weight:800;color:#fff;">
              {d["best_model_name"].replace("_"," ").title()}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:11px;color:rgba(255,255,255,.45);letter-spacing:.06em;
                        text-transform:uppercase;margin-bottom:4px;">{d["primary_metric"].upper()}</div>
            <div style="font-size:15px;font-weight:800;color:#7DDFF5;">{roc_hero:.3f}</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Detail-level badge ───────────────────────────────────────────────────
    lv = st.session_state.detail_level
    lv_colors = {"business": GREEN, "analyst": ORANGE, "technical": BLUE}
    lv_labels = {"business": "普通用户 · 业务语言", "analyst": "分析师 · 加入关键指标", "technical": "技术细节 · 完整数据"}
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:7px;
                background:{lv_colors[lv]}14;border:1px solid {lv_colors[lv]}44;
                border-radius:20px;padding:5px 14px;margin-bottom:18px;
                font-size:11px;color:{lv_colors[lv]};font-weight:700;letter-spacing:.03em;">
      <div style="width:6px;height:6px;border-radius:50%;background:{lv_colors[lv]};"></div>
      当前模式：{lv_labels[lv]}
    </div>""", unsafe_allow_html=True)

    # ── Chat history ─────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:12px;">'
                    f'<div class="msg-user">{msg["text"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                text = msg["text"]
                text_html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                text_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_html)
                text_html = re.sub(r'`(.+?)`', r'<code>\1</code>', text_html)
                text_html = text_html.replace("\n\n", "<br><br>").replace("\n", "<br>")
                st.markdown(
                    f'<div style="display:flex;margin-bottom:12px;">'
                    f'<div class="msg-bot">{text_html}</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Quick-action grid ────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{SUB};margin:8px 0 10px;">快捷问题</div>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "这份数据有什么问题？",
        "模型结果怎么样？",
        "哪些变量最重要？",
        "模型可以上线吗？",
        "下一步应该做什么？",
        "为什么是分类/回归？",
        "数据是怎么清洗的？",
        "模型排行榜",
    ]
    pill_cols = st.columns(4)
    for i, sug in enumerate(suggestions):
        with pill_cols[i % 4]:
            if st.button(sug, key=f"pill_{i}"):
                st.session_state.chat_history.append({"role": "user", "text": sug})
                answer = _answer(sug, d)
                if st.session_state.detail_level == "analyst":
                    bm = d["bm"]; pm = d["primary_metric"]
                    rv = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
                    answer += (f"\n\n📊 **补充指标：** {pm.upper()}={rv:.3f}  "
                               f"F1={bm.get('test_f1',0):.3f}  Acc={bm.get('test_accuracy',0):.3f}")
                elif st.session_state.detail_level == "technical":
                    answer += "\n\n🔬 技术细节请切换到 **技术仪表板** 查看完整诊断报告。"
                st.session_state.chat_history.append({"role": "bot", "text": answer})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat input ───────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "输入您的问题…",
                label_visibility="collapsed",
                placeholder="例如：这个模型准吗？有什么风险？",
            )
        with col_btn:
            submitted = st.form_submit_button("发送 ↵")

    if submitted and user_input.strip():
        q = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "text": q})
        answer = _answer(q, d)
        if st.session_state.detail_level == "analyst":
            bm = d["bm"]; pm = d["primary_metric"]
            rv = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
            answer += (f"\n\n📊 **补充指标：** {pm.upper()}={rv:.3f}  "
                       f"F1={bm.get('test_f1',0):.3f}  Acc={bm.get('test_accuracy',0):.3f}")
        elif st.session_state.detail_level == "technical":
            answer += "\n\n🔬 技术细节请切换到 **技术仪表板** 查看完整诊断报告。"
        st.session_state.chat_history.append({"role": "bot", "text": answer})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ██  VIEW — SUBMIT JOB
# ─────────────────────────────────────────────────────────────────────────────
elif "提交" in view:

    st.markdown(f"""
    <div class="hero-banner" style="background:linear-gradient(135deg,{NAVY} 0%,{BRIGHT} 100%);">
      <div style="font-size:22px;font-weight:900;letter-spacing:-.5px;margin-bottom:6px;">
        提交新分析任务
      </div>
      <div style="font-size:13px;color:rgba(255,255,255,.65);">
        上传数据集，描述业务目标 — AutoDS 自动完成全流程分析
      </div>
    </div>""", unsafe_allow_html=True)

    # Step 1
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <div style="width:28px;height:28px;border-radius:50%;
                  background:linear-gradient(135deg,{NAVY},{BLUE});
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">1</div>
      <div style="font-size:15px;font-weight:700;color:{NAVY};">上传数据集</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("选择 CSV 文件", type=["csv"], label_visibility="collapsed")

    if uploaded:
        preview_df = pd.read_csv(uploaded)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:10px 0 8px;">'
            f'<div class="stage-pill">✓ 已上传</div>'
            f'<div style="font-size:12px;color:{SUB};">'
            f'{uploaded.name} · {len(preview_df):,} 行 · {len(preview_df.columns)} 列</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(preview_df.head(6), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 2
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <div style="width:28px;height:28px;border-radius:50%;
                  background:linear-gradient(135deg,{NAVY},{BLUE});
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">2</div>
      <div style="font-size:15px;font-weight:700;color:{NAVY};">描述业务目标</div>
    </div>""", unsafe_allow_html=True)

    business_desc = st.text_area(
        "业务描述",
        placeholder="例：我们希望预测客户是否会流失。目标列是 churn，希望提高召回率以减少漏报。",
        height=110,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Step 3
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <div style="width:28px;height:28px;border-radius:50%;
                  background:linear-gradient(135deg,{NAVY},{BLUE});
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">3</div>
      <div style="font-size:15px;font-weight:700;color:{NAVY};">目标列 & 任务类型</div>
    </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        if uploaded:
            opts = ["（让 AutoDS 自动推断）"] + preview_df.columns.tolist()
            sel = st.selectbox("目标列", opts)
            target_col = "" if "自动" in sel else sel
        else:
            target_col = st.text_input("目标列（留空让 AutoDS 推断）", placeholder="例如 target")
    with cb:
        prob_sel = st.selectbox("任务类型", ["自动判断", "分类（Classification）", "回归（Regression）"])
        prob_map = {
            "自动判断": "classification",
            "分类（Classification）": "classification",
            "回归（Regression）": "regression",
        }
        problem_type = prob_map[prob_sel]

    st.markdown("<br>", unsafe_allow_html=True)

    can_run = uploaded is not None
    if not can_run:
        st.markdown(
            f'<div style="font-size:12px;color:{SUB};padding:8px 0;">请先上传 CSV 文件。</div>',
            unsafe_allow_html=True,
        )

    run_btn = st.button("▶  运行 AutoDS 流水线", disabled=not can_run)

    if run_btn and can_run:
        tmp = PROJECT_ROOT / "_uploaded_data_temp.csv"
        tmp.write_bytes(uploaded.getvalue())
        with st.spinner("🔄 正在运行分析流水线，请稍候…"):
            ok, err = run_pipeline(str(tmp), business_desc, target_col.strip(), problem_type)
        if ok:
            st.session_state.chat_history = []
            get_pipeline_data.clear()
            st.success("✅ 分析完成！切换到 **对话分析** 开始提问。")
            st.balloons()
        else:
            st.error("流水线执行失败，详见错误信息：")
            st.code(err, language="python")


# ─────────────────────────────────────────────────────────────────────────────
# ██  VIEW — TECHNICAL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif "仪表板" in view:

    if not results_exist:
        st.markdown(f"""
        <div style="text-align:center;padding:100px 0 80px;max-width:480px;margin:0 auto;">
          <div style="font-size:52px;margin-bottom:20px;">📭</div>
          <div style="font-size:22px;font-weight:800;color:{NAVY};margin-bottom:10px;
                      letter-spacing:-.4px;">暂无分析结果</div>
          <div style="font-size:14px;color:{SUB};line-height:1.6;">请先提交任务运行流水线。</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    d  = get_pipeline_data()
    bm = d["bm"]; pm = d["primary_metric"]; pt = d["problem_type"]
    bmn = d["best_model_name"]; lb = d["lb_list"]
    roc  = bm.get("test_roc_auc") or bm.get(f"test_{pm}", 0)
    f1   = bm.get("test_f1", 0)
    acc  = bm.get("test_accuracy", 0)
    prec = bm.get("test_precision", 0)
    rec  = bm.get("test_recall", 0)
    feat_imp = d["feat_imp"]

    # Dashboard sub-nav in sidebar
    with st.sidebar:
        st.markdown('<div style="height:1px;background:rgba(255,255,255,.10);margin:12px 0 14px;"></div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:9px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;'
            'color:rgba(255,255,255,.28);padding:0 2px 8px;">流水线阶段</div>',
            unsafe_allow_html=True,
        )
        dash_nav = st.radio(
            "DashNav",
            ["⬡  总览", "◎  数据理解", "◈  数据清洗", "◆  特征工程", "◉  建模", "◐  评估", "📄  报告"],
            label_visibility="collapsed",
        )

    # ──────── Total overview ──────────────────────────────────────────────────
    if "总览" in dash_nav:
        st.markdown(f"""
        <div class="hero-banner">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
            <div>
              <div style="font-size:22px;font-weight:900;letter-spacing:-.5px;">流水线总览</div>
              <div style="font-size:12px;color:rgba(255,255,255,.5);margin-top:5px;">
                运行时间：{d["timestamp"]}
              </div>
            </div>
            <div style="background:rgba(27,138,86,.2);border:1px solid rgba(27,138,86,.4);
                        border-radius:12px;padding:12px 20px;text-align:center;">
              <div style="font-size:28px;font-weight:900;color:#7DDFF5;">{roc:.3f}</div>
              <div style="font-size:10px;color:rgba(255,255,255,.5);letter-spacing:.06em;
                          text-transform:uppercase;margin-top:3px;">{pm.upper()} 最优</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Pipeline steps
        steps = [
            ("01", "数据理解", "Schema + 质量"),
            ("02", "数据清洗", "去重 · 填充"),
            ("03", "特征工程", "选择 · 缩放"),
            ("04", "建模",     f"{len(lb)} 候选模型"),
            ("05", "评估",     "排行榜"),
            ("06", "报告",     "MD + JSON"),
        ]
        step_cols = st.columns(len(steps))
        for col, (num, name, sub) in zip(step_cols, steps):
            col.markdown(f"""
            <div class="step-card">
              <div class="step-check">✓</div>
              <div class="step-name">{name}</div>
              <div class="step-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI row
        clean_sum = d["clean_sum"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("原始样本",    d["n_rows_in"],
                  f"{d['in_shape'].get('columns','—')} 原始特征")
        k2.metric("数据保留率",  f"{clean_sum.get('data_retention_percentage',0):.0f}%",
                  f"移除 {clean_sum.get('rows_removed',0)} 行")
        k3.metric("最终特征数",  d["final_feat_cnt"], "特征选择后")
        k4.metric("候选模型数",  len(lb), f"最优：{bmn.replace('_',' ').title()}")
        k5.metric(f"最优 {pm.upper()}", f"{roc:.3f}", "测试集")

    # ──────── Data understanding ──────────────────────────────────────────────
    elif "数据理解" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            01 · 数据理解
          </div>
        </div>""", unsafe_allow_html=True)

        dh = d["understanding"].get("downstream_handoff", {})
        fe = dh.get("feature_engineering_agent", {})
        mo = dh.get("modelling_agent", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div style="font-size:10px;font-weight:800;color:{SUB};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">数据集规模</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame({
                "指标": ["行数", "列数", "数值列", "类别列"],
                "值": [
                    d["n_rows_in"], d["in_shape"].get("columns", "—"),
                    len(fe.get("numeric_columns", [])), len(fe.get("categorical_columns", [])),
                ],
            }), hide_index=True, use_container_width=True)
        with c2:
            st.markdown(
                f'<div style="font-size:10px;font-weight:800;color:{SUB};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">目标列</div>',
                unsafe_allow_html=True,
            )
            imb_ratio = "—"
            for f in d["major_findings"]:
                m = re.search(r'\d+\.\d+', f)
                if m and ("imbalanc" in f.lower() or "ratio" in f.lower()):
                    imb_ratio = m.group(); break
            st.dataframe(pd.DataFrame({
                "项目": ["目标列", "任务类型", "不平衡比例", "主评估指标"],
                "值": [
                    mo.get("target_column", "—"),
                    mo.get("problem_type", "—").title(),
                    imb_ratio, pm.upper(),
                ],
            }), hide_index=True, use_container_width=True)
        with c3:
            st.markdown(
                f'<div style="font-size:10px;font-weight:800;color:{SUB};text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">质量检查</div>',
                unsafe_allow_html=True,
            )
            dqm = d["cleaning"].get("data_quality_metrics", {}).get("original", {})
            ids = fe.get("suspected_identifier_columns", [])
            st.dataframe(pd.DataFrame({
                "检查项": ["缺失值", "重复行", "常量列", "ID-like 列"],
                "状态": [
                    "✓ 无" if dqm.get("null_values", 0) == 0 else f"⚠ {dqm.get('null_values')}",
                    f"✓ {dqm.get('duplicates', 0)}",
                    "✓ 0",
                    f"⚠ {len(ids)} 个" if ids else "✓ 无",
                ],
            }), hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:12px;">主要发现</div>',
            unsafe_allow_html=True,
        )
        for finding in d["major_findings"]:
            is_warn = "imbalance" in finding.lower() or "identifier" in finding.lower()
            dot_color = ORANGE if is_warn else BLUE
            st.markdown(f"""
            <div class="finding-row">
              <div class="finding-dot" style="background:{dot_color};"></div>
              <div style="font-size:13px;color:#1E2D3D;line-height:1.5;">{finding}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:12px;">主要风险</div>',
                unsafe_allow_html=True,
            )
            for r in d["primary_risks"]:
                st.markdown(f"""
                <div style="display:flex;gap:10px;padding:10px 14px;background:#FEF3E2;
                            border:1px solid #F8D89C;border-radius:10px;margin-bottom:8px;
                            font-size:13px;color:#7C4A00;align-items:flex-start;">
                  <span style="font-size:15px;">⚠</span>
                  <span style="line-height:1.5;">{r}</span>
                </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:12px;">建议后续步骤</div>',
                unsafe_allow_html=True,
            )
            for i, s in enumerate(d["next_steps"]):
                st.markdown(f"""
                <div style="display:flex;gap:12px;padding:10px 14px;background:#EAF3FB;
                            border:1px solid #B8D8F0;border-radius:10px;margin-bottom:8px;
                            font-size:13px;color:#0E2F63;align-items:flex-start;">
                  <div style="width:22px;height:22px;border-radius:50%;
                              background:{BLUE};color:#fff;font-size:11px;font-weight:800;
                              display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                    {i+1}
                  </div>
                  <span style="line-height:1.5;">{s}</span>
                </div>""", unsafe_allow_html=True)

    # ──────── Cleaning ────────────────────────────────────────────────────────
    elif "清洗" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            02 · 数据清洗
          </div>
        </div>""", unsafe_allow_html=True)

        cs = d["clean_sum"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("输入行数",  d["n_rows_in"],  f"{d['in_shape'].get('columns','—')} 列")
        k2.metric("输出行数",  d["n_rows_out"], f"{d['out_shape'].get('columns','—')} 列")
        k3.metric("移除行数",  cs.get("rows_removed", 0), "异常/重复")
        ret = float(cs.get("data_retention_percentage", 0) or 0)
        k4.metric("保留率", f"{ret:.1f}%",
                  "✓ 优秀" if ret >= 95 else "⚠ 偏低")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:12px;">清洗前后对比</div>',
                unsafe_allow_html=True,
            )
            dqo = d["cleaning"].get("data_quality_metrics", {}).get("original", {})
            dqp = d["cleaning"].get("data_quality_metrics", {}).get("processed", {}) or {}
            st.dataframe(pd.DataFrame({
                "指标":  ["完整率 (%)", "重复行", "空值", "异常行"],
                "清洗前": [
                    dqo.get("completeness", 100), dqo.get("duplicates", 0),
                    dqo.get("null_values", 0), cs.get("rows_removed", 0),
                ],
                "清洗后": [
                    dqp.get("completeness", 100), dqp.get("duplicates", 0),
                    dqp.get("null_values", 0), 0,
                ],
            }), hide_index=True, use_container_width=True)
        with c2:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:12px;">执行的操作</div>',
                unsafe_allow_html=True,
            )
            ops = [
                ("🔤", "列名标准化",      "snake_case"),
                ("🗑", f"移除重复行",     f"{dqo.get('duplicates',0)} 行"),
                ("🔢", "缺失值填充",      "中位数填充"),
                ("📐", "IQR 异常值过滤",  f"移除 {cs.get('rows_removed',0)} 行"),
            ]
            for icon, label, tag in ops:
                st.markdown(f"""
                <div class="op-row">
                  <span style="font-size:16px;">{icon}</span>
                  <span style="flex:1;font-weight:600;color:#24364B;">{label}</span>
                  <span class="stat-badge">{tag}</span>
                </div>""", unsafe_allow_html=True)

    # ──────── Feature engineering ─────────────────────────────────────────────
    elif "特征" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            03 · 特征工程
          </div>
        </div>""", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("最终特征数",  d["final_feat_cnt"], "特征选择后")
        k2.metric("训练样本",    d["n_train"] or "—", "~80%")
        k3.metric("测试样本",    d["n_test"] or "—",  "~20%")
        k4.metric("交叉验证折数", 5, "分层 K-fold")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">特征重要性排名</div>',
                unsafe_allow_html=True,
            )
            fi = feat_imp
            if not fi.empty:
                mx = fi["importance"].max()
                for i, row in fi.iterrows():
                    p = row["importance"] / mx * 100 if mx else 0
                    st.markdown(f"""
                    <div class="feat-bar">
                      <div class="feat-bar-top">
                        <span class="feat-name">{row['feature_name']}</span>
                        <span class="feat-rank">#{i+1} · {int(row['importance'])}</span>
                      </div>
                      <div class="feat-track">
                        <div class="feat-fill" style="width:{p:.0f}%;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("暂无特征重要性数据。")
        with c2:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">训练/测试划分</div>',
                unsafe_allow_html=True,
            )
            nt, nv = d["n_train"], d["n_test"]
            if nt and nv:
                fig = go.Figure(go.Pie(
                    labels=[f"训练集 ({nt})", f"测试集 ({nv})"],
                    values=[nt, nv],
                    hole=0.70,
                    marker=dict(colors=[BLUE, CYAN], line=dict(color="#fff", width=3)),
                    hovertemplate="%{label}: %{value} 样本<extra></extra>",
                ))
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1, font=dict(size=12, color=SUB)),
                    margin=dict(t=10, b=20, l=0, r=0),
                    height=240,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ──────── Modelling ───────────────────────────────────────────────────────
    elif "建模" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            04 · 建模
          </div>
        </div>""", unsafe_allow_html=True)

        lbl, _ = score_label(roc)

        # Champion card
        st.markdown(f"""
        <div class="hero-banner" style="margin-bottom:24px;">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:22px;">
            <div style="width:48px;height:48px;background:rgba(255,255,255,.12);
                        border-radius:13px;display:flex;align-items:center;
                        justify-content:center;font-size:24px;border:1px solid rgba(255,255,255,.15);">
              🏆
            </div>
            <div style="flex:1;">
              <div style="font-size:20px;font-weight:900;letter-spacing:-.3px;">
                {bmn.replace("_"," ").title()}
              </div>
              <div style="font-size:11px;color:rgba(255,255,255,.5);margin-top:3px;">
                最优模型 · 排名 #1 by {pm.upper()} · 共 {len(lb)} 个候选
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:36px;font-weight:900;color:#7DDFF5;letter-spacing:-1px;">
                {roc:.3f}
              </div>
              <div style="font-size:10px;color:rgba(255,255,255,.45);letter-spacing:.05em;">
                测试集 {pm.upper()}
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            {''.join(f"""
            <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);
                        border-radius:10px;padding:14px;text-align:center;">
              <div style="font-size:22px;font-weight:900;letter-spacing:-.5px;">{v:.3f}</div>
              <div style="font-size:10px;color:rgba(255,255,255,.45);margin-top:4px;
                          text-transform:uppercase;letter-spacing:.06em;">{lbl2}</div>
            </div>""" for v, lbl2 in [(acc,"Accuracy"),(prec,"Precision"),(rec,"Recall"),(f1,"F1")])}
          </div>
        </div>""", unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">测试集指标</div>',
                unsafe_allow_html=True,
            )
            for label, val in [
                ("ROC-AUC", roc), ("F1", f1), ("Recall", rec), ("Precision", prec), ("Accuracy", acc)
            ]:
                st.markdown(f"""
                <div class="mbar-wrap">
                  <div class="mbar-label">{label}</div>
                  <div class="mbar-track">
                    <div class="mbar-fill" style="width:{val*100:.1f}%;"></div>
                  </div>
                  <div class="mbar-val">{val:.3f}</div>
                </div>""", unsafe_allow_html=True)

        with mc2:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">雷达图 — 前两名模型</div>',
                unsafe_allow_html=True,
            )
            if len(lb) >= 2:
                cats = ["ROC-AUC", "F1", "Recall", "Precision", "Accuracy"]
                fig = go.Figure()
                for m, color, fill in zip(lb[:2], [BLUE, CYAN],
                                          ["rgba(35,97,174,.18)", "rgba(59,165,199,.12)"]):
                    vals = [
                        m["test_roc_auc"] * 100, m["test_f1"] * 100,
                        m["test_recall"] * 100, m["test_precision"] * 100,
                        m["test_accuracy"] * 100,
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=cats + [cats[0]],
                        fill="toself", fillcolor=fill,
                        line=dict(color=color, width=2),
                        name=m["model_name"].replace("_", " ").title(),
                        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
                    ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[30, 100], showticklabels=False,
                                        gridcolor=BORDER, linecolor=BORDER),
                        angularaxis=dict(gridcolor=BORDER, linecolor=BORDER),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.14, font=dict(size=11, color=SUB)),
                    margin=dict(t=10, b=40, l=20, r=20), height=270,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        if not feat_imp.empty:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin:20px 0 14px;">特征重要性</div>',
                unsafe_allow_html=True,
            )
            palette = [NAVY, BLUE, BRIGHT, CYAN, MIST] * 4
            fig = go.Figure(go.Bar(
                x=feat_imp["importance"],
                y=feat_imp["feature_name"],
                orientation="h",
                marker=dict(color=palette[:len(feat_imp)], cornerradius=5),
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            fig.update_layout(
                xaxis=dict(gridcolor=BORDER, tickfont=dict(color=SUB, size=11)),
                yaxis=dict(showgrid=False, tickfont=dict(color="#24364B", size=11)),
                margin=dict(t=10, b=10, l=0, r=0),
                height=max(120, len(feat_imp) * 44),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin:20px 0 14px;">模型排行榜</div>',
            unsafe_allow_html=True,
        )
        if not d["leaderboard"].empty:
            lb_df = d["leaderboard"].copy()
            show_cols = [c for c in [
                "rank", "model_name", "test_roc_auc", "test_f1",
                "test_accuracy", "test_precision", "test_recall",
                "cv_roc_auc", "cv_runtime_seconds",
            ] if c in lb_df.columns]
            display = lb_df[show_cols].copy()
            display.columns = [c.replace("_", " ").title() for c in show_cols]
            for col in display.columns:
                if display[col].dtype == float:
                    display[col] = display[col].round(4)
            st.dataframe(display, hide_index=True, use_container_width=True)

    # ──────── Evaluation ──────────────────────────────────────────────────────
    elif "评估" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            05 · 评估
          </div>
        </div>""", unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">ROC-AUC 对比</div>',
                unsafe_allow_html=True,
            )
            if lb:
                names   = [m["model_name"].replace("_", " ").title() for m in lb]
                palette = [NAVY, BLUE, BRIGHT, CYAN, MIST]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="测试集", x=names,
                    y=[m["test_roc_auc"] for m in lb],
                    marker=dict(color=palette[:len(names)], cornerradius=6),
                    hovertemplate="%{x}: %{y:.3f}<extra>Test</extra>",
                ))
                fig.add_trace(go.Bar(
                    name="CV 均值", x=names,
                    y=[m["cv_roc_auc"] for m in lb],
                    marker=dict(color=f"rgba(35,97,174,.14)",
                                line=dict(color=f"rgba(35,97,174,.35)", width=1.5),
                                cornerradius=6),
                    hovertemplate="%{x}: %{y:.3f}<extra>CV</extra>",
                ))
                fig.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 1], gridcolor=BORDER, tickfont=dict(color=SUB)),
                    xaxis=dict(showgrid=False, tickfont=dict(color="#24364B", size=11)),
                    legend=dict(orientation="h", y=-0.2, font=dict(size=11, color=SUB)),
                    margin=dict(t=10, b=56, l=0, r=0), height=270,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        with ec2:
            st.markdown(
                f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin-bottom:14px;">多指标对比</div>',
                unsafe_allow_html=True,
            )
            if lb:
                names = [m["model_name"].replace("_", " ").title() for m in lb]
                fig = go.Figure()
                for mname, key, mc in [
                    ("Accuracy",  "test_accuracy",  f"rgba(14,54,124,.85)"),
                    ("Precision", "test_precision", f"rgba(35,97,174,.85)"),
                    ("Recall",    "test_recall",    f"rgba(43,124,197,.85)"),
                    ("F1",        "test_f1",        f"rgba(59,165,199,.85)"),
                ]:
                    fig.add_trace(go.Bar(
                        name=mname, x=names, y=[m[key] for m in lb],
                        marker=dict(color=mc, cornerradius=4),
                        hovertemplate=f"{mname} — %{{x}}: %{{y:.3f}}<extra></extra>",
                    ))
                fig.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 1], gridcolor=BORDER, tickfont=dict(color=SUB)),
                    xaxis=dict(showgrid=False, tickfont=dict(color="#24364B", size=10)),
                    legend=dict(orientation="h", y=-0.2, font=dict(size=11, color=SUB)),
                    margin=dict(t=10, b=56, l=0, r=0), height=270,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        se1, se2, se3 = st.columns(3)
        se1.metric("选定模型",   bmn.replace("_", " ").title())
        se2.metric("选择指标",   pm.upper())
        se3.metric("测试集得分", f"{roc:.3f}")

        st.markdown(
            f'<div style="font-size:14px;font-weight:800;color:{NAVY};margin:24px 0 14px;">部署建议</div>',
            unsafe_allow_html=True,
        )
        lbl, _ = score_label(roc)
        ids = (d["understanding"].get("downstream_handoff", {})
               .get("feature_engineering_agent", {})
               .get("suspected_identifier_columns", []))
        recs = [
            ("green", "✅", f"部署 {bmn.replace('_',' ').title()} 到测试环境",
             "模型在测试集和交叉验证中均表现稳定。"),
            ("warn",  "⚠",  "监控数据分布漂移",
             "随着新数据积累定期重训练。"),
            ("blue",  "↗",  "收集更多标注数据",
             f"当前 {pm.upper()}={roc:.3f}，仍有提升空间。"),
        ]
        if ids:
            recs.append(("navy", "◎", "排查 ID-like 列泄漏风险",
                          f"{', '.join(ids)} 疑似为标识符列，上线前确认无数据泄漏。"))
        alert_styles = {
            "green": (GREEN,  "#E8F7EE", "#A8D5B5", "#1a5c38"),
            "warn":  (ORANGE, "#FEF3E2", "#F8D89C", "#7c4a00"),
            "blue":  (BLUE,   "#EAF3FB", "#B8D8F0", "#0e2f63"),
            "navy":  (NAVY,   "#E8EEF6", "#C2CDE0", "#0a1f46"),
        }
        for key, icon, title, desc in recs:
            _, bg, border, text_col = alert_styles[key]
            st.markdown(f"""
            <div class="alert-row" style="background:{bg};border:1px solid {border};">
              <div class="alert-icon">{icon}</div>
              <div>
                <div class="alert-title" style="color:{text_col};">{title}</div>
                <div class="alert-desc" style="color:{SUB};">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ──────── Report ──────────────────────────────────────────────────────────
    elif "报告" in dash_nav:
        st.markdown(f"""
        <div style="padding-bottom:20px;border-bottom:2px solid {BORDER};margin-bottom:24px;">
          <div style="font-size:22px;font-weight:900;color:{NAVY};letter-spacing:-.5px;">
            06 · 报告
          </div>
        </div>""", unsafe_allow_html=True)

        rt1, rt2 = st.tabs(["📋 业务报告", "🔬 技术报告"])
        with rt1:
            br = d.get("business_report", "")
            if br:
                st.markdown(br)
            else:
                st.info("暂无业务报告。")
        with rt2:
            tr = d["report"].get("technical", "")
            if tr:
                st.markdown(tr)
            else:
                st.info("暂无技术报告。")
