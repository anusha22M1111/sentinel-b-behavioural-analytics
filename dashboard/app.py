import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="SENTINEL-B Command Center")

FEATURES = [
    "logins","delay","attendance","sentiment",
    "night_activity_ratio","engagement_velocity",
    "engagement_acceleration","attendance_trend",
    "sentiment_drift","volatility_index",
    "burnout_risk_score"
]

ACCENT = "#00d4ff"
ORANGE = "#ff7b00"
GREEN = "#00ff88"
RED = "#ff3b3b"

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE/"data"/"synthetic_student_behavior.csv")
model = joblib.load(BASE/"models"/"trained_model.pkl")

df["student_name"] = "Student_" + df["student_id"].astype(str)

mode = st.sidebar.radio("Select View",
                        ["Individual Intelligence",
                         "Institutional Command Center"])

# ============================================================
# INDIVIDUAL INTELLIGENCE
# ============================================================
if mode == "Individual Intelligence":

    st.title("🎓 Student Behavioural Intelligence")

    student = st.selectbox("Select Student",
                           sorted(df["student_name"].unique()))

    student_df = df[df["student_name"] == student].sort_values("week")
    latest = student_df.iloc[-1]

    # KPI STRIP
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk Score", round(latest["burnout_risk_score"],2))
    c2.metric("Dropout %", round(latest["dropout_probability"]*100,2))
    c3.metric("Attendance %", round(latest["attendance"],2))
    c4.metric("Sentiment", round(latest["sentiment"],2))

    # ROW 1
    r1 = st.columns(4)

    # Risk Timeline
    with r1[0]:
        fig = px.line(student_df,
                      x="week",
                      y="burnout_risk_score",
                      color_discrete_sequence=[ACCENT],
                      title="Risk Timeline")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Risk Acceleration
    with r1[1]:
        student_df["risk_acceleration"] = student_df["burnout_risk_score"].diff()
        fig = px.line(student_df,
                      x="week",
                      y="risk_acceleration",
                      color_discrete_sequence=[ORANGE],
                      title="Risk Acceleration")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Risk Gauge
    with r1[2]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest["burnout_risk_score"],
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":ORANGE},
                "steps":[
                    {"range":[0,40],"color":GREEN},
                    {"range":[40,70],"color":"yellow"},
                    {"range":[70,100],"color":RED}
                ]
            }))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # State Distribution
    with r1[3]:
        fig = px.bar(student_df["state"].value_counts(),
                     color_discrete_sequence=[ACCENT],
                     title="State Frequency")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ROW 2
    r2 = st.columns(4)

    # Transition Heatmap
    with r2[0]:
        trans = pd.crosstab(student_df["state"],
                            student_df["state"].shift(-1))
        fig = px.imshow(trans,
                        color_continuous_scale="Blues",
                        title="Transition Matrix")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    with r2[1]:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(FEATURES))

        imp_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig = px.bar(imp_df,
                     x="Importance",
                     y="Feature",
                     orientation="h",
                     color_discrete_sequence=[ORANGE],
                     title="Feature Importance")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Radar Profile
    with r2[2]:
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(student_df[FEATURES])
        avg_profile = norm.mean(axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg_profile,
            theta=FEATURES,
            fill="toself"
        ))
        fig.update_layout(template="plotly_dark",
                          height=300,
                          title="Behavioural Radar")
        st.plotly_chart(fig, use_container_width=True)

    # AI Insight
    with r2[3]:
        st.subheader("Behavioural Insight")
        trend = "increasing" if student_df["burnout_risk_score"].iloc[-1] > student_df["burnout_risk_score"].iloc[0] else "stable"
        st.write(f"• Risk trajectory is **{trend}**.")
        st.write("• Primary driver appears to be engagement volatility.")
        st.write("• Recommend proactive intervention if risk exceeds 60.")

# ============================================================
# INSTITUTIONAL COMMAND CENTER
# ============================================================
else:

    st.title("🏛 Institutional Behavioural Command Center")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Students", df["student_id"].nunique())
    c2.metric("Avg Risk", round(df["burnout_risk_score"].mean(),2))
    c3.metric("High Risk %",
              round((df["risk_level"]=="High").mean()*100,2))
    c4.metric("Avg Dropout %",
              round(df["dropout_probability"].mean()*100,2))

    r1 = st.columns(4)

    with r1[0]:
        fig = px.histogram(df,
                           x="burnout_risk_score",
                           nbins=40,
                           title="Risk Distribution",
                           color_discrete_sequence=[ACCENT])
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with r1[1]:
        weekly = df.groupby("week")["burnout_risk_score"].mean()
        fig = px.line(weekly,
                      title="Weekly Avg Risk",
                      color_discrete_sequence=[ORANGE])
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with r1[2]:
        fig = px.bar(df["state"].value_counts(),
                     title="State Distribution",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with r1[3]:
        high_risk = df[df["risk_level"]=="High"]
        st.subheader("Top High Risk Students")
        top = high_risk.groupby("student_name")["burnout_risk_score"].max().sort_values(ascending=False).head(10)
        st.dataframe(top)

    r2 = st.columns(2)

    with r2[0]:
        trans = pd.crosstab(df["state"],
                            df["state"].shift(-1))
        fig = px.imshow(trans,
                        title="Global Transition Matrix",
                        color_continuous_scale="Blues")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with r2[1]:
        fig = px.histogram(df,
                           x="dropout_probability",
                           nbins=40,
                           title="Dropout Probability Distribution",
                           color_discrete_sequence=[ORANGE])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

st.success("SENTINEL-B Behavioural Intelligence System Active")