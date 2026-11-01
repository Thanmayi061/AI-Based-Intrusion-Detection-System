import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Smart AI Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Auto Refresh every 2 seconds
st_autorefresh(interval=2000, key="refresh")

# =====================================================
# Title
# =====================================================

st.title("🛡️ Smart AI Intrusion Detection System")
st.markdown("### Live Network Traffic Monitoring Dashboard")

LOG_FILE = "logs/predictions.csv"

# =====================================================
# Check Log File
# =====================================================

if not os.path.exists(LOG_FILE):
    st.error("Prediction log not found.")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.warning("Prediction log is empty.")
    st.stop()

# =====================================================
# Statistics
# =====================================================

total_packets = len(df)
benign_packets = len(df[df["Prediction"] == "BENIGN"])
attack_packets = total_packets - benign_packets

col1, col2, col3 = st.columns(3)

col1.metric("📦 Total Packets", total_packets)
col2.metric("✅ Benign", benign_packets)
col3.metric("⚠️ Attacks", attack_packets)

st.divider()

# =====================================================
# Latest Status
# =====================================================

latest = df.iloc[-1]

if latest["Prediction"] == "BENIGN":

    st.success("🟢 NETWORK STATUS : SAFE")

else:

    st.error("🚨 ATTACK DETECTED")

    st.write(f"### Attack Type : {latest['Prediction']}")
    st.write(f"Source IP : {latest['Source IP']}")
    st.write(f"Destination IP : {latest['Destination IP']}")
    st.write(f"Protocol : {latest['Protocol']}")
    st.write(f"Confidence : {latest['Confidence (%)']} %")

st.divider()

# =====================================================
# Bar Charts
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Attack Distribution")

    st.bar_chart(df["Prediction"].value_counts())

with col2:

    st.subheader("🌐 Protocol Distribution")

    st.bar_chart(df["Protocol"].value_counts())

st.divider()

# =====================================================
# Pie Charts
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🥧 Attack Distribution")

    attack_counts = df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.pie(
        attack_counts,
        labels=attack_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.65,
        textprops={"fontsize": 7}
    )

    ax.axis("equal")

    st.pyplot(fig)

with col2:

    st.subheader("🥧 Protocol Distribution")

    protocol_counts = df["Protocol"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(3, 3))

    ax2.pie(
        protocol_counts,
        labels=protocol_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.65,
        textprops={"fontsize": 7}
    )

    ax2.axis("equal")

    st.pyplot(fig2)

st.divider()
# =====================================================
# Donut Charts
# =====================================================

col1, col2 = st.columns(2)

# ---------------- Attack Donut Chart ---------------- #

with col1:

    st.subheader("🍩 Attack Distribution")

    attack_counts = df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(3,3))

    ax.pie(
        attack_counts,
        labels=attack_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.9,
        wedgeprops=dict(width=0.35, edgecolor="white"),
        textprops={'fontsize':8}
    )

    ax.set(aspect="equal")

    st.pyplot(fig)

# ---------------- Protocol Donut Chart ---------------- #

with col2:

    st.subheader("🍩 Protocol Distribution")

    protocol_counts = df["Protocol"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(3,3))

    ax2.pie(
        protocol_counts,
        labels=protocol_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        radius=0.9,
        wedgeprops=dict(width=0.35, edgecolor="white"),
        textprops={'fontsize':8}
    )

    ax2.set(aspect="equal")

    st.pyplot(fig2)

st.divider()

# =====================================================
# Detection History
# =====================================================

st.subheader("📈 Detection History")

history = df["Prediction"].value_counts().reset_index()
history.columns = ["Prediction", "Count"]

st.line_chart(history.set_index("Prediction"))

st.divider()

# =====================================================
# Recent Predictions
# =====================================================

st.subheader("📋 Recent Predictions")

st.dataframe(df.tail(20), use_container_width=True)

st.divider()

# =====================================================
# Download CSV
# =====================================================

st.download_button(
    label="📥 Download Prediction Log",
    data=df.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
)

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.markdown(
    """
    <center>
    <h4>🛡️ Smart AI-Based Network Intrusion Detection System</h4>
    Developed using <b>Python • Scapy • Machine Learning • Streamlit</b>
    </center>
    """,
    unsafe_allow_html=True
)