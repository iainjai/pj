# st.py — Presentation Edition (Single Prediction Only)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============== UI ==============
st.set_page_config(page_title="Injury Risk — Next Season", page_icon="🩺", layout="wide")
st.markdown("<h1 style='text-align:center'>🩺 Injury Risk — Next Season</h1>", unsafe_allow_html=True)

MODEL_PATH = "models/Injury_Next_Season_model.joblib"
POSITIONS = ["Defender", "Midfielder", "Forward", "Goalkeeper"]

NUMERIC = ["Age","Height_cm","Weight_kg","Training_Hours_Per_Week",
           "Matches_Played_Past_Season","Previous_Injury_Count",
           "Knee_Strength_Score","Hamstring_Flexibility","Reaction_Time_ms",
           "Balance_Test_Score","Sprint_Speed_10m_s","Agility_Score",
           "Sleep_Hours_Per_Night","Stress_Level_Score",
           "Nutrition_Quality_Score","Warmup_Routine_Adherence","BMI"]

@st.cache_resource(show_spinner=False)
def load_pipeline(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    if hasattr(obj, "predict"):
        return obj
    raise ValueError("Unsupported model artifact.")

def sanitize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Position" in df.columns:
        df["Position"] = df["Position"].astype(str).fillna("Unknown")
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)

def align_columns(df: pd.DataFrame, feature_names_in: list | None) -> pd.DataFrame:
    if feature_names_in is None:
        return df
    X = df.copy()
    for c in feature_names_in:
        if c not in X.columns:
            X[c] = 0
    return X.reindex(columns=feature_names_in)

def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    return model.predict(X).astype(float)

def gauge(proba: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Risk", "font": {"size": 20}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "red" if proba >= 0.5 else "green"}}
    ))
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=300)
    st.plotly_chart(fig, use_container_width=True)

pipe = load_pipeline(MODEL_PATH)
feature_names_in = getattr(pipe, "feature_names_in_", None)

# ============== Inputs (compact) ==============
st.markdown("### Enter Player Data")
c1, c2, c3, c4 = st.columns(4)

age    = c1.number_input("Age", 15, 45, 25)
height = c1.number_input("Height_cm", 150, 210, 180)
weight = c1.number_input("Weight_kg", 45, 120, 75)
bmi    = c1.number_input("BMI", 10.0, 45.0, round(weight/((height/100)**2), 2))

thpw   = c2.number_input("Training_Hours_Per_Week", 0.0, 30.0, 10.0)
matches= c2.number_input("Matches_Played_Past_Season", 0, 60, 20)
prev_i = c2.number_input("Previous_Injury_Count", 0, 20, 1)
warmup = c2.selectbox("Warmup_Routine_Adherence", [0, 1], 1)

knee   = c3.number_input("Knee_Strength_Score", 0.0, 100.0, 75.0)
hams   = c3.number_input("Hamstring_Flexibility", 0.0, 100.0, 80.0)
react  = c3.number_input("Reaction_Time_ms", 150.0, 400.0, 250.0)
balance= c3.number_input("Balance_Test_Score", 0.0, 100.0, 85.0)

sprint = c4.number_input("Sprint_Speed_10m_s", 4.5, 8.0, 6.0)
agility= c4.number_input("Agility_Score", 0.0, 100.0, 75.0)
sleep  = c4.number_input("Sleep_Hours_Per_Night", 3.0, 12.0, 7.5)
stress = c4.number_input("Stress_Level_Score", 0.0, 100.0, 50.0)

pos    = st.selectbox("Position", POSITIONS, 1)

# ============== Predict Button ==============
center = st.container()
with center:
    if st.button("Predict", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Age": age, "Height_cm": height, "Weight_kg": weight,
            "Training_Hours_Per_Week": thpw, "Matches_Played_Past_Season": matches,
            "Previous_Injury_Count": prev_i, "Knee_Strength_Score": knee,
            "Hamstring_Flexibility": hams, "Reaction_Time_ms": react,
            "Balance_Test_Score": balance, "Sprint_Speed_10m_s": sprint,
            "Agility_Score": agility, "Sleep_Hours_Per_Night": sleep,
            "Stress_Level_Score": stress, "Nutrition_Quality_Score": 70.0,
            "Warmup_Routine_Adherence": warmup, "BMI": bmi, "Position": pos
        }])
        row = sanitize_inputs(row)
        X_single = align_columns(row, feature_names_in)
        proba = float(predict_proba_safe(pipe, X_single)[0])
        pred  = int(proba >= 0.5)

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("<h3>Risk Probability</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin-top:-10px'>{proba*100:.1f}%</h2>", unsafe_allow_html=True)
        with col2:
            gauge(proba)

        st.markdown(
            f"<div style='text-align:center; font-size:22px; padding:8px; "
            f"border-radius:10px; background:{'#ffe5e5' if pred==1 else '#e6ffea'}'>"
            f"Predicted class: <b>{pred}</b></div>",
            unsafe_allow_html=True
        )

