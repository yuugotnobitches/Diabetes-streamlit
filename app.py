import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="DiabetesIQ — Risk Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0d0f12;
}

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 980px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #141619; }
::-webkit-scrollbar-thumb { background: #2a2d35; border-radius: 3px; }

/* ── Hero ── */
.hero {
    position: relative;
    border: 1px solid #1e2128;
    border-radius: 20px;
    padding: 52px 52px 44px;
    margin-bottom: 28px;
    background: #111317;
    overflow: hidden;
}
.hero-dot-grid {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(#1e2430 1px, transparent 1px);
    background-size: 24px 24px;
    opacity: 0.7;
}
.hero-glow {
    position: absolute;
    top: -120px; right: -80px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,150,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #00c896;
    margin-bottom: 14px;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 42px;
    font-weight: 700;
    color: #edf0f4;
    line-height: 1.15;
    margin: 0 0 16px;
    letter-spacing: -0.02em;
}
.hero h1 span { color: #00c896; }
.hero p {
    font-size: 15px;
    color: #6b7280;
    line-height: 1.7;
    max-width: 520px;
    margin: 0 0 28px;
}
.pill-row { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 5px 13px;
    border-radius: 20px;
    border: 1px solid #252830;
    color: #8b95a3;
    background: #161820;
}

/* ── Stats ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.stat-card {
    background: #111317;
    border: 1px solid #1e2128;
    border-radius: 14px;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c896, transparent);
}
.stat-num {
    font-size: 28px;
    font-weight: 700;
    color: #edf0f4;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-num span { color: #00c896; }
.stat-desc { font-size: 12px; color: #4b5563; line-height: 1.5; }

/* ── Section label ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d4450;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2128;
}

/* ── Card ── */
.card {
    background: #111317;
    border: 1px solid #1e2128;
    border-radius: 16px;
    padding: 26px 26px 22px;
    margin-bottom: 18px;
}

/* ── Inputs ── */
.stNumberInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4b5563 !important;
    margin-bottom: 6px !important;
}
.stNumberInput input {
    background: #0d0f12 !important;
    border: 1px solid #252830 !important;
    border-radius: 8px !important;
    color: #d1d5db !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.stNumberInput input:focus {
    border-color: #00c896 !important;
    box-shadow: 0 0 0 2px rgba(0,200,150,0.08) !important;
}
/* number input buttons */
.stNumberInput button {
    background: #1a1d24 !important;
    border-color: #252830 !important;
    color: #4b5563 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: #00c896 !important;
    color: #0a0d10 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 15px 24px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00e6ad !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,200,150,0.2) !important;
}

/* ── Result cards ── */
.result-box {
    border-radius: 16px;
    padding: 26px 28px;
    margin-top: 18px;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: radial-gradient(#ffffff08 1px, transparent 1px);
    background-size: 20px 20px;
}
.result-safe   { background: #0b1f17; border: 1px solid #1a4030; }
.result-warn   { background: #1c1a0b; border: 1px solid #3d3410; }
.result-danger { background: #1f0e0e; border: 1px solid #3d1515; }

.result-inner { position: relative; z-index: 1; }
.result-risk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.safe-accent   { color: #00c896; }
.warn-accent   { color: #f0b429; }
.danger-accent { color: #f56565; }

.result-title {
    font-size: 22px;
    font-weight: 700;
    color: #edf0f4;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.result-sub {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 20px;
    line-height: 1.6;
}
.gauge-wrap { margin-bottom: 6px; }
.gauge-bg {
    height: 6px;
    border-radius: 3px;
    background: #1e2128;
    overflow: hidden;
}
.gauge-fill-safe   { height:100%; border-radius:3px; background: linear-gradient(90deg,#00a878,#00c896); }
.gauge-fill-warn   { height:100%; border-radius:3px; background: linear-gradient(90deg,#d4a017,#f0b429); }
.gauge-fill-danger { height:100%; border-radius:3px; background: linear-gradient(90deg,#c53030,#f56565); }
.gauge-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #3d4450;
    margin-top: 5px;
}

/* ── Recommendations ── */
.rec-section {
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid #1e2128;
    position: relative;
    z-index: 1;
}
.rec-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 14px;
}
.rec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.rec-card {
    border-radius: 10px;
    padding: 13px 15px;
    border: 1px solid;
}
.rec-safe   { background: #0d201a; border-color: #1a4030; }
.rec-warn   { background: #1c1b0d; border-color: #3d3410; }
.rec-danger { background: #1e1010; border-color: #3d1515; }

.rec-icon { font-size: 18px; margin-bottom: 6px; }
.rec-title {
    font-size: 12px;
    font-weight: 600;
    color: #c9d1da;
    margin-bottom: 3px;
}
.rec-body {
    font-size: 11px;
    color: #4b5563;
    line-height: 1.5;
}

/* ── Right panel cards ── */
.info-card {
    background: #111317;
    border: 1px solid #1e2128;
    border-radius: 16px;
    padding: 22px 24px 20px;
    margin-bottom: 16px;
}
.step-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 16px;
}
.step-row:last-child { margin-bottom: 0; }
.step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid #252830;
    background: #161820;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00c896;
}
.step-body h4 { font-size: 13px; font-weight: 600; color: #c9d1da; margin: 0 0 3px; }
.step-body p  { font-size: 12px; color: #4b5563; margin: 0; line-height: 1.5; }

.factor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.factor-tile {
    background: #0d0f12;
    border: 1px solid #1e2128;
    border-radius: 10px;
    padding: 12px 14px;
}
.factor-name { font-size: 12px; font-weight: 600; color: #8b95a3; margin-bottom: 4px; }
.factor-desc { font-size: 11px; color: #3d4450; line-height: 1.5; }

/* ── Disclaimer ── */
.disclaimer {
    background: #131208;
    border: 1px solid #2a2610;
    border-radius: 10px;
    padding: 13px 16px;
    margin-top: 4px;
}
.disclaimer p {
    font-size: 12px;
    color: #6b6030;
    line-height: 1.5;
    margin: 0;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #111317;
    border: 1px solid #1e2128;
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 10px !important; color: #3d4450 !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { font-size: 18px !important; color: #c9d1da !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #2a2d35;
    padding: 28px 0 8px;
    border-top: 1px solid #1a1d22;
    margin-top: 36px;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = joblib.load("diabetes_model.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

model, scaler = load_model()

# ── Recommendations by tier ─────────────────────────────────────────────────
def get_recommendations(tier):
    if tier == "safe":
        return [
            ("🥗", "Balanced diet", "Prioritise whole grains, legumes, and vegetables. Limit refined sugars and processed carbs."),
            ("🏃", "Regular exercise", "Aim for 150 min/week of moderate aerobic activity. Walking, cycling or swimming all count."),
            ("💧", "Stay hydrated", "Drink 2–3L of water daily. Avoid sugary drinks and limit fruit juice intake."),
            ("📅", "Annual screening", "Keep up with yearly glucose and HbA1c checks even with a low-risk profile."),
        ]
    elif tier == "warn":
        return [
            ("🩸", "Monitor glucose", "Check fasting blood glucose every 3–6 months. Ask your doctor about HbA1c testing."),
            ("⚖️", "Weight management", "Losing 5–7% of body weight significantly reduces progression to Type 2 diabetes."),
            ("🚫", "Cut refined carbs", "Replace white rice, bread, and sugar with low-GI alternatives like oats and legumes."),
            ("🧘", "Reduce stress", "Chronic stress raises cortisol and blood sugar. Yoga, meditation, or therapy can help."),
            ("🩺", "See your doctor", "Request a formal glucose tolerance test and discuss your family history with a physician."),
            ("🚶", "Walk after meals", "A 10-minute post-meal walk can reduce blood sugar spikes by up to 22%."),
        ]
    else:
        return [
            ("💊", "Medical consultation", "Consult an endocrinologist or diabetologist immediately for a formal diagnosis and care plan."),
            ("📋", "HbA1c test", "Get a glycated haemoglobin (HbA1c) test. Values ≥ 6.5% confirm Type 2 diabetes diagnosis."),
            ("🍽️", "Medical nutrition therapy", "Work with a registered dietitian for a structured meal plan — carb counting is key."),
            ("💉", "Medication review", "Depending on severity, your doctor may prescribe Metformin or other glucose-lowering agents."),
            ("🫀", "Cardiovascular check", "Diabetes significantly raises heart risk. Monitor blood pressure, LDL cholesterol, and kidney function."),
            ("👁️", "Eye & foot exams", "Annual retinal and podiatric exams detect early diabetic complications — neuropathy and retinopathy."),
        ]

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-dot-grid"></div>
  <div class="hero-glow"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">// CSE × Biotech Research &nbsp;·&nbsp; ML-powered clinical tool</div>
    <h1>Diabetes <span>Risk</span><br>Detection System</h1>
    <p>Enter a patient's clinical measurements to receive a machine-learning risk
       assessment — plus personalised preventive or curative recommendations
       based on your result.</p>
    <div class="pill-row">
      <span class="pill">Random Forest</span>
      <span class="pill">Pima Dataset</span>
      <span class="pill">StandardScaler</span>
      <span class="pill">Early Detection</span>
      <span class="pill">Actionable Advice</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
  <div class="stat-card">
    <div class="stat-num">537<span>M+</span></div>
    <div class="stat-desc">adults living with diabetes worldwide (IDF 2021)</div>
  </div>
  <div class="stat-card">
    <div class="stat-num"><span>~</span>77<span>%</span></div>
    <div class="stat-desc">typical Random Forest accuracy on Pima dataset</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">768</div>
    <div class="stat-desc">patient records used to train the model</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">8</div>
    <div class="stat-desc">clinical features used for inference</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ─────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Patient details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose     = st.number_input("Glucose mg/dL", 0, 200, 120)
        bp          = st.number_input("Blood Pressure mmHg", 0, 150, 70)
        skin        = st.number_input("Skin Thickness mm", 0, 100, 20)
    with c2:
        insulin = st.number_input("Insulin μU/mL", 0, 900, 80)
        bmi     = st.number_input("BMI kg/m²", 0.0, 70.0, 25.0, step=0.1)
        dpf     = st.number_input("Pedigree Function", 0.0, 2.5, 0.50, step=0.01)
        age     = st.number_input("Age years", 1, 120, 30)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN RISK ASSESSMENT →"):
        input_data   = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)
        probability  = model.predict_proba(input_scaled)[0][1]
        pct          = int(probability * 100)

        # Determine tier
        if pct < 35:
            tier = "safe"
        elif pct < 65:
            tier = "warn"
        else:
            tier = "danger"

        tier_meta = {
            "safe":   ("safe-accent",  "result-safe",  "LOW RISK",      "✓ Low Diabetes Risk Detected",         f"The model places this patient at {100-pct}% likelihood of being diabetes-free. Continue preventive habits."),
            "warn":   ("warn-accent",  "result-warn",  "MODERATE RISK", "⚡ Moderate Risk — Action Advised",     f"Risk score of {pct}% suggests pre-diabetic indicators. Lifestyle changes can significantly reduce progression."),
            "danger": ("danger-accent","result-danger", "HIGH RISK",    "⚠ High Risk — Seek Medical Attention",  f"Risk score of {pct}% indicates strong diabetic markers. Immediate clinical evaluation is strongly recommended."),
        }
        accent, box_cls, risk_tag, title, sub = tier_meta[tier]

        fill_cls = {"safe":"gauge-fill-safe","warn":"gauge-fill-warn","danger":"gauge-fill-danger"}[tier]
        fill_pct = pct if tier != "safe" else 100 - pct

        recs = get_recommendations(tier)
        rec_card_cls = {"safe":"rec-safe","warn":"rec-warn","danger":"rec-danger"}[tier]

        rec_html = ""
        for icon, rtitle, rbody in recs:
            rec_html += f"""
            <div class="rec-card {rec_card_cls}">
              <div class="rec-icon">{icon}</div>
              <div class="rec-title">{rtitle}</div>
              <div class="rec-body">{rbody}</div>
            </div>"""

        rec_cols = f'<div class="rec-grid">{rec_html}</div>'

        st.markdown(f"""
        <div class="result-box {box_cls}">
          <div class="result-inner">
            <div class="result-risk-label {accent}">{risk_tag}</div>
            <div class="result-title">{title}</div>
            <div class="result-sub">{sub}</div>
            <div class="gauge-wrap">
              <div class="gauge-bg"><div class="{fill_cls}" style="width:{fill_pct}%"></div></div>
              <div class="gauge-meta"><span>0%</span><span style="color:#6b7280">{fill_pct}% risk score</span><span>100%</span></div>
            </div>
            <div class="rec-section">
              <div class="rec-heading">
                {'— Preventive measures' if tier == 'safe' else '— Early intervention steps' if tier == 'warn' else '— Curative & management steps'}
              </div>
              {rec_cols}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Input summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Glucose", f"{glucose}")
        m2.metric("BMI", f"{bmi:.1f}")
        m3.metric("Age", f"{age}")
        m4.metric("Insulin", f"{insulin}")

with right:
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
      <div class="step-row">
        <div class="step-circle">01</div>
        <div class="step-body">
          <h4>Enter clinical values</h4>
          <p>Fill in the 8 diagnostic measurements from a standard patient workup.</p>
        </div>
      </div>
      <div class="step-row">
        <div class="step-circle">02</div>
        <div class="step-body">
          <h4>Feature scaling</h4>
          <p>A pre-fitted StandardScaler normalises all inputs so the model can compare them on equal footing.</p>
        </div>
      </div>
      <div class="step-row">
        <div class="step-circle">03</div>
        <div class="step-body">
          <h4>ML inference</h4>
          <p>A Random Forest classifier trained on 614 Pima patient records outputs a binary prediction + probability.</p>
        </div>
      </div>
      <div class="step-row">
        <div class="step-circle">04</div>
        <div class="step-body">
          <h4>Tiered recommendations</h4>
          <p>Based on the risk score, you receive targeted preventive or curative action steps.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Risk tiers explained</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
      <div style="display:flex;flex-direction:column;gap:12px">
        <div style="display:flex;align-items:center;gap:14px;padding:12px 14px;background:#0b1f17;border:1px solid #1a4030;border-radius:10px">
          <div style="width:8px;height:8px;border-radius:50%;background:#00c896;flex-shrink:0"></div>
          <div>
            <div style="font-size:12px;font-weight:600;color:#00c896;margin-bottom:2px">Low Risk &nbsp;·&nbsp; 0–34%</div>
            <div style="font-size:11px;color:#4b5563">Healthy indicators. Focus on maintaining habits and annual screening.</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:14px;padding:12px 14px;background:#1c1a0b;border:1px solid #3d3410;border-radius:10px">
          <div style="width:8px;height:8px;border-radius:50%;background:#f0b429;flex-shrink:0"></div>
          <div>
            <div style="font-size:12px;font-weight:600;color:#f0b429;margin-bottom:2px">Moderate Risk &nbsp;·&nbsp; 35–64%</div>
            <div style="font-size:11px;color:#4b5563">Pre-diabetic signals present. Lifestyle changes are highly effective at this stage.</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:14px;padding:12px 14px;background:#1f0e0e;border:1px solid #3d1515;border-radius:10px">
          <div style="width:8px;height:8px;border-radius:50%;background:#f56565;flex-shrink:0"></div>
          <div>
            <div style="font-size:12px;font-weight:600;color:#f56565;margin-bottom:2px">High Risk &nbsp;·&nbsp; 65–100%</div>
            <div style="font-size:11px;color:#4b5563">Strong diabetic markers. Immediate medical consultation and diagnostic tests required.</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Why these factors matter</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
      <div class="factor-grid">
        <div class="factor-tile"><div class="factor-name">🩸 Glucose</div><div class="factor-desc">Primary biomarker of impaired insulin regulation and pre-diabetes onset.</div></div>
        <div class="factor-tile"><div class="factor-name">⚖️ BMI</div><div class="factor-desc">Higher BMI strongly correlates with insulin resistance and T2D risk.</div></div>
        <div class="factor-tile"><div class="factor-name">🧬 Pedigree</div><div class="factor-desc">Encodes hereditary risk from relatives with a diabetes diagnosis.</div></div>
        <div class="factor-tile"><div class="factor-name">🎂 Age</div><div class="factor-desc">Metabolic efficiency declines with age, raising insulin dysfunction risk.</div></div>
        <div class="factor-tile"><div class="factor-name">💉 Insulin</div><div class="factor-desc">2-hr serum level reveals how the body responds to a glucose load.</div></div>
        <div class="factor-tile"><div class="factor-name">🫀 BP</div><div class="factor-desc">Hypertension frequently co-occurs and compounds cardiovascular risk.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <p>⚠️ <strong style="color:#8b7d30">Clinical disclaimer:</strong> This is a research prototype for educational purposes only and is not a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified clinician.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  DIABETESIQ &nbsp;·&nbsp; CSE &amp; BIOTECH RESEARCH TEAM &nbsp;·&nbsp;
  PIMA INDIANS DIABETES DATASET (UCI ML REPOSITORY) &nbsp;·&nbsp;
  NOT FOR CLINICAL USE
</div>
""", unsafe_allow_html=True)