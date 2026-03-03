import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="ReviewShield",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("ReviewShield")
st.markdown("**Fake Product Review Detection System** — Powered by XGBoost + ML")
st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Review Checker", "📦 Bulk CSV Analysis"])


# ════════════════════════════════════════════════════════
# TAB 1 — SINGLE REVIEW
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Enter Review Details")
        review_text = st.text_area(
            "Review Text",
            placeholder="Paste the product review here...",
            height=180
        )
        rating = st.slider("Star Rating", min_value=1, max_value=5, value=5)
        analyze_btn = st.button("🔍 Analyze Review", use_container_width=True)

    with col2:
        if analyze_btn:
            if not review_text.strip():
                st.warning("Please enter a review first.")
            else:
                with st.spinner("Analyzing..."):
                    response = requests.post(f"{API_URL}/predict", json={
                        "text": review_text,
                        "rating": rating
                    })

                if response.status_code == 200:
                    data = response.json()

                    # ── Verdict ──────────────────────────
                    if data["prediction"] == "Fake":
                        st.error(f" **FAKE REVIEW DETECTED**")
                    else:
                        st.success(f" **GENUINE REVIEW**")

                    # ── Confidence Gauge ─────────────────
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=data["confidence"],
                        title={"text": f"Confidence — {data['risk_level']} Risk"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "crimson" if data["prediction"] == "Fake" else "green"},
                            "steps": [
                                {"range": [0, 50], "color": "#e8f5e9"},
                                {"range": [50, 75], "color": "#fff9c4"},
                                {"range": [75, 100], "color": "#ffebee"},
                            ]
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Explanation ───────────────────────
                    st.info(f" {data['explanation']}")

                    # ── Feature Breakdown ─────────────────
                    st.subheader("Feature Breakdown")
                    feat = data["features_used"]
                    f_col1, f_col2 = st.columns(2)
                    f_col1.metric("Review Length", f"{feat['review_length']} chars")
                    f_col1.metric("Avg Sentence Length", f"{round(feat['avg_sentence_length'], 1)}")
                    f_col2.metric("Unique Word Ratio", f"{round(feat['unique_word_ratio']*100, 1)}%")
                    f_col2.metric("Contains Numbers", "Yes ✅" if feat['has_digits'] else "No ❌")

                else:
                    st.error(f"API Error: {response.text}")


# ════════════════════════════════════════════════════════
# TAB 2 — BULK CSV
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    st.markdown("CSV must have two columns: `text` and `rating`")

    # Download sample CSV button
    sample_df = pd.DataFrame({
        "text": [
            "Amazing product! Best ever! Love it so much!!!",
            "I bought this for my home office. Build quality is solid. Cable management decent. Instructions unclear.",
            "Great quality! Perfect! Highly recommend to everyone!",
            "Battery lasts about 3 hours. Came with USB-C charger. Good value for 1500 rupees."
        ],
        "rating": [5, 4, 5, 3]
    })
    st.download_button(
        "Download Sample CSV",
        sample_df.to_csv(index=False),
        "sample_reviews.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns or "rating" not in df.columns:
            st.error("CSV must have 'text' and 'rating' columns.")
        else:
            st.write(f"Loaded {len(df)} reviews. Preview:")
            st.dataframe(df.head())

            if st.button("Run Bulk Analysis", use_container_width=True):
                reviews = df[["text", "rating"]].to_dict(orient="records")

                # Send in batches of 100