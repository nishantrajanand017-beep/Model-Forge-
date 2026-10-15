"""Visual theming and components for Model Forge: a restrained, professional
black-and-white design system with clean typography and subtle borders.
No gradients, no neon colors, no emojis."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ---- Typography & Base Reset ---- */
html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #18181B;
    background-color: #FAFAF9 !important;
}
code, pre, kbd {
    font-family: ui-monospace, "Cascadia Code", "SFMono-Regular", Menlo, Monaco, Consolas, monospace !important;
}

/* Material icons support */
.stApp [data-testid="stIconMaterial"],
.stApp .material-symbols-rounded,
.stApp .material-symbols-outlined,
.stApp .material-icons {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}

/* ---- Streamlit Chrome Clean-up ---- */
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stToolbar"], [data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stDeployButton"], .stDeployButton { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }

/* ---- Main Content Card ---- */
.block-container {
    background: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 8px !important;
    padding: 2.25rem 2.5rem 3.5rem 2.5rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 2.5rem !important;
    max-width: 1060px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
}

/* ---- Headings ---- */
h1, h2, h3, h4, h5, h6 {
    color: #18181B !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}
h1 { font-size: 1.45rem !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.15rem !important; margin-top: 1.25rem !important; margin-bottom: 0.4rem !important; }
h3 { font-size: 0.98rem !important; margin-top: 0.9rem !important; margin-bottom: 0.35rem !important; }
p:not(button *), span:not(button *), label:not(button *) { color: #27272A; }

/* ---- Header Bar ---- */
.mf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #E4E4E7;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
}
.mf-brand {
    display: flex;
    flex-direction: column;
}
.mf-wordmark {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #18181B;
    text-transform: uppercase;
}
.mf-sub {
    font-size: 0.825rem;
    color: #71717A;
    margin-top: 0.15rem;
    font-weight: 400;
}
.mf-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.mf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #3F3F46;
    background: #F4F4F5;
    border: 1px solid #E4E4E7;
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
}
.mf-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #18181B;
}
.mf-badge-idle {
    color: #71717A;
    background: #FAFAFA;
}
.mf-badge-dot-idle {
    background: #A1A1AA;
}

/* ---- Section Headers ---- */
.mf-section {
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    padding-top: 1.25rem;
    border-top: 1px solid #F4F4F5;
}
.mf-section:first-of-type {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}
.mf-section-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}
.mf-section-tag {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #71717A;
}
.mf-section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #18181B;
}
.mf-section-sub {
    font-size: 0.825rem;
    color: #71717A;
    margin-bottom: 0.9rem;
    line-height: 1.45;
}

/* ---- Primary Action Buttons (e.g., Run Unsupervised / Run Supervised) ---- */
.stButton > button,
button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: #18181B !important;
    background-color: #18181B !important;
    color: #FFFFFF !important;
    border: 1px solid #18181B !important;
    border-radius: 6px !important;
    min-height: 46px !important;
    height: 46px !important;
    padding: 0.65rem 1.75rem !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.5 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: none !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}

/* Ensure text and all nested elements inside primary button are pure white */
.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div,
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
button[data-testid="baseButton-primary"] *,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] div {
    color: #FFFFFF !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
}

/* Primary Button Hover */
.stButton > button:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: #27272A !important;
    background-color: #27272A !important;
    border-color: #27272A !important;
    color: #FFFFFF !important;
}
.stButton > button:hover *,
button[kind="primary"]:hover * {
    color: #FFFFFF !important;
}

/* Primary Button Focus */
.stButton > button:focus,
.stButton > button:focus-visible,
button[kind="primary"]:focus,
button[kind="primary"]:focus-visible,
button[data-testid="baseButton-primary"]:focus,
button[data-testid="baseButton-primary"]:focus-visible {
    background: #18181B !important;
    background-color: #18181B !important;
    border-color: #18181B !important;
    color: #FFFFFF !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #18181B !important;
}
.stButton > button:focus *,
button[kind="primary"]:focus * {
    color: #FFFFFF !important;
}

/* Primary Button Active */
.stButton > button:active,
button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active {
    background: #09090B !important;
    background-color: #09090B !important;
    border-color: #09090B !important;
    color: #FFFFFF !important;
    transform: translateY(1px);
}
.stButton > button:active *,
button[kind="primary"]:active * {
    color: #FFFFFF !important;
}

/* Primary Button Disabled */
.stButton > button:disabled,
.stButton > button[disabled],
button[kind="primary"]:disabled,
button[kind="primary"][disabled],
button[data-testid="baseButton-primary"]:disabled,
button[data-testid="baseButton-primary"][disabled] {
    background: #F4F4F5 !important;
    background-color: #F4F4F5 !important;
    border: 1px solid #D4D4D8 !important;
    color: #52525B !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
    box-shadow: none !important;
    transform: none !important;
}
.stButton > button:disabled *,
.stButton > button[disabled] *,
button[kind="primary"]:disabled *,
button[kind="primary"][disabled] *,
button[data-testid="baseButton-primary"]:disabled *,
button[data-testid="baseButton-primary"][disabled] * {
    color: #52525B !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* ---- Download / Secondary Buttons ---- */
.stDownloadButton > button,
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #18181B !important;
    border: 1px solid #D4D4D8 !important;
    border-radius: 6px !important;
    min-height: 44px !important;
    height: 44px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
.stDownloadButton > button *,
button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] * {
    color: #18181B !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover,
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: #F4F4F5 !important;
    background-color: #F4F4F5 !important;
    border-color: #A1A1AA !important;
    color: #18181B !important;
}
.stDownloadButton > button:hover *,
button[kind="secondary"]:hover * {
    color: #18181B !important;
}
.stDownloadButton > button:focus,
.stDownloadButton > button:focus-visible,
button[kind="secondary"]:focus,
button[kind="secondary"]:focus-visible {
    background: #FFFFFF !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px #D4D4D8 !important;
}
.stDownloadButton > button:active,
button[kind="secondary"]:active {
    background: #E4E4E7 !important;
    transform: translateY(1px);
}


/* ---- Form Inputs, Selects, Sliders ---- */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    color: #18181B !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #A1A1AA !important;
}
.stTextInput input, .stNumberInput input {
    background-color: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    color: #18181B !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #18181B !important;
    box-shadow: none !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    margin-top: 0.25rem;
}

/* Mode Selector segmented appearance */
.stRadio [data-testid="stWidgetLabel"] {
    display: none !important;
}
.stRadio > div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    background: #F4F4F5;
    padding: 0.25rem 0.35rem;
    border-radius: 7px;
    border: 1px solid #E4E4E7;
    width: fit-content;
    margin-bottom: 1rem;
}
.stRadio > div[role="radiogroup"] label {
    margin: 0 !important;
    padding: 0.3rem 0.85rem !important;
    border-radius: 5px !important;
    cursor: pointer !important;
    font-size: 0.825rem !important;
    font-weight: 500 !important;
    color: #52525B !important;
    transition: all 0.15s ease !important;
}
.stRadio > div[role="radiogroup"] label:hover {
    color: #18181B !important;
}
.stRadio > div[role="radiogroup"] label[data-checked="true"],
.stRadio > div[role="radiogroup"] label:has(input:checked) {
    background: #FFFFFF !important;
    color: #18181B !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06) !important;
}

/* Radio circle inputs minimal styling */
.stRadio input[type="radio"] {
    accent-color: #18181B !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #E4E4E7 !important;
    gap: 1.25rem !important;
}
.stTabs [data-baseweb="tab"] {
    color: #71717A !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.15rem !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #18181B !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #18181B !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #18181B !important;
    height: 2px !important;
}

/* ---- Metric Cards ---- */
[data-testid="stMetric"] {
    background: #FAFAFA !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stMetricLabel"] {
    color: #71717A !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricValue"] {
    color: #18181B !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    border: 1px solid #E4E4E7 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* ---- File Uploader ---- */
[data-testid="stFileUploader"] {
    border: 1px dashed #D4D4D8 !important;
    border-radius: 6px !important;
    padding: 1.25rem !important;
    background: #FAFAF9 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #18181B !important;
}

/* ---- MultiSelect Tags ---- */
span[data-baseweb="tag"] {
    background-color: #F4F4F5 !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
}
span[data-baseweb="tag"] span {
    color: #18181B !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
span[data-baseweb="tag"] svg {
    fill: #71717A !important;
}
span[data-baseweb="tag"]:hover svg {
    fill: #18181B !important;
}

/* ---- Alerts / Notices ---- */
.stAlert, [data-testid="stAlert"] {
    border-radius: 6px !important;
    border: 1px solid #E4E4E7 !important;
    border-left: 3px solid #18181B !important;
    background-color: #FAFAFA !important;
    color: #18181B !important;
    font-size: 0.85rem !important;
}
div[data-testid="stAlert"] > div, [data-testid="stAlert"] p, [data-testid="stAlert"] span {
    color: #18181B !important;
}
[data-testid="stAlert"] svg {
    fill: #18181B !important;
}

/* ---- Empty State Notice ---- */
.mf-empty-state {
    padding: 2.5rem 1.5rem;
    text-align: center;
    border: 1px dashed #E4E4E7;
    border-radius: 6px;
    background: #FAFAFA;
    margin: 1rem 0;
}
.mf-empty-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #18181B;
    margin-bottom: 0.25rem;
}
.mf-empty-sub {
    font-size: 0.825rem;
    color: #71717A;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ---- Decision Flow Cards ---- */
.dec-group-title {
    font-weight: 600;
    color: #18181B;
    font-size: 0.95rem;
    margin: 1.25rem 0 0.4rem 0;
}
.dec-why {
    background: #FAFAFA;
    border: 1px solid #E4E4E7;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.dec-why-item {
    margin-bottom: 0.5rem;
}
.dec-why-item:last-child {
    margin-bottom: 0;
}
.dec-why-head {
    color: #18181B;
    font-weight: 600;
    font-size: 0.85rem;
}
.dec-why-head .pick {
    color: #18181B;
    text-decoration: underline;
    font-weight: 700;
}
.dec-why-text {
    color: #52525B;
    font-size: 0.8rem;
    line-height: 1.45;
    margin-top: 0.1rem;
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(dataset_name: str | None = None, n_rows: int | None = None, n_cols: int | None = None):
    """Render the clean application header with wordmark, subtitle, and dataset badge."""
    if dataset_name and n_rows is not None and n_cols is not None:
        status_html = (
            f'<div class="mf-badge">'
            f'<span class="mf-badge-dot"></span>'
            f'<span>{dataset_name} · {n_rows:,} rows · {n_cols:,} cols</span>'
            f'</div>'
        )
    else:
        status_html = (
            '<div class="mf-badge mf-badge-idle">'
            '<span class="mf-badge-dot mf-badge-dot-idle"></span>'
            '<span>No dataset loaded</span>'
            '</div>'
        )

    html = (
        f'<div class="mf-header">'
        f'<div class="mf-brand">'
        f'<div class="mf-wordmark">Model Forge</div>'
        f'<div class="mf-sub">Automated Machine Learning</div>'
        f'</div>'
        f'<div class="mf-status">{status_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section(tag: str, title: str, subtitle: str | None = None):
    """Render a clean section divider and heading."""
    sub_html = f'<div class="mf-section-sub">{subtitle}</div>' if subtitle else ""
    html = (
        f'<div class="mf-section">'
        f'<div class="mf-section-header">'
        f'<span class="mf-section-tag">{tag}</span>'
        f'<span class="mf-section-title">{title}</span>'
        f'</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(title: str = "No dataset loaded",
                       message: str = "Upload a CSV or Excel file to preview data and configure model analysis."):
    """Render a clean, quiet empty state."""
    st.markdown(
        f'<div class="mf-empty-state">'
        f'<div class="mf-empty-title">{title}</div>'
        f'<div class="mf-empty-sub">{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_hero():
    """Backward compatibility hook."""
    pass


def render_flowchart():
    """Backward compatibility hook for initial empty state."""
    render_empty_state(
        title="Upload a dataset to begin",
        message="Support for CSV and Excel spreadsheets (.csv, .xlsx, .xls). Select Supervised mode to predict a target column or Unsupervised mode to discover clusters, anomalies, and lower-dimensional structures."
    )


def _decisions_to_dot(decisions, root_label):
    """Build a Graphviz DOT tree using monochrome styling:
    solid black for chosen nodes/arrows, clean light borders for unchosen options."""
    CHOSEN_FILL = "#18181B"
    CHOSEN_TEXT = "#FFFFFF"
    DIM_FILL = "#FFFFFF"
    DIM_BORDER = "#D4D4D8"
    DIM_TEXT = "#71717A"
    EDGE_CHOSEN = "#18181B"
    EDGE_DIM = "#D4D4D8"
    out = [
        "digraph {",
        'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
        'node [shape=box, style="rounded,filled", fontname="Segoe UI", '
        'fontsize=11, margin="0.18,0.09", penwidth=1.0];',
        "edge [arrowsize=0.7];",
        f'root [label="{root_label}", fillcolor="{CHOSEN_FILL}", color="{CHOSEN_FILL}", fontcolor="{CHOSEN_TEXT}", penwidth=1.5];',
    ]
    prev = "root"
    for li, d in enumerate(decisions):
        ids = []
        chosen_id = None
        for oi, (label, chosen) in enumerate(d["options"]):
            nid = f"n{li}_{oi}"
            ids.append(nid)
            safe = str(label).replace('"', "'")
            if chosen:
                out.append(
                    f'{nid} [label="{safe}", fillcolor="{CHOSEN_FILL}", color="{CHOSEN_FILL}", '
                    f'fontcolor="{CHOSEN_TEXT}", penwidth=1.5];'
                )
                chosen_id = nid
            else:
                out.append(
                    f'{nid} [label="{safe}", fillcolor="{DIM_FILL}", '
                    f'color="{DIM_BORDER}", fontcolor="{DIM_TEXT}", penwidth=1.0];'
                )
        for nid, (label, chosen) in zip(ids, d["options"]):
            if chosen:
                out.append(f'{prev} -> {nid} [color="{EDGE_CHOSEN}", penwidth=1.8];')
            else:
                out.append(f'{prev} -> {nid} [color="{EDGE_DIM}", penwidth=1.0];')
        out.append("{ rank=same; " + " ".join(ids) + " }")
        prev = chosen_id or (ids[0] if ids else prev)
    out.append("}")
    return "\n".join(out)


def render_decision_graph(title, decisions, root_label):
    if not decisions:
        return
    st.markdown(f'<div class="dec-group-title">{title}</div>', unsafe_allow_html=True)
    st.graphviz_chart(_decisions_to_dot(decisions, root_label), use_container_width=True)
    items = []
    for d in decisions:
        picks = [label for label, chosen in d["options"] if chosen]
        pick = ", ".join(picks) if picks else "n/a"
        items.append(
            '<div class="dec-why-item">'
            f'<div class="dec-why-head">{d["question"]}: <span class="pick">{pick}</span></div>'
            f'<div class="dec-why-text">{d["reason"]}</div>'
            "</div>"
        )
    st.markdown('<div class="dec-why">' + "".join(items) + "</div>", unsafe_allow_html=True)

