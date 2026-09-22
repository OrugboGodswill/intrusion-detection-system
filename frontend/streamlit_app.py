import base64
import io
import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("IDS_API_URL", "http://localhost:5000")
st.set_page_config(page_title="Sentinel IDS", page_icon="S", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container { max-width: 1280px; padding-top: 2rem; }
.hero { padding: 2rem; border: 1px solid #263348; background: linear-gradient(135deg,#101a2d,#0b1220); border-radius: 18px; }
.eyebrow { color:#67e8f9; letter-spacing:.14em; text-transform:uppercase; font-size:.72rem; font-weight:700; }
.metric { border:1px solid #263348; border-radius:14px; padding:1rem; background:#111b2d; }
.status { padding:.7rem 1rem; border-radius:10px; background:#2a1d0d; color:#fbbf24; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">Security operations / network intelligence</div><h1>Sentinel IDS</h1><p>Batch intrusion detection with explainable Random Forest predictions.</p></div>', unsafe_allow_html=True)

health = requests.get(f"{API_URL}/api/health", timeout=5).json()
if health["status"] != "ready":
    st.markdown('<div class="status">Model artifact not connected. Add <b>models/random_forest_pipeline4.joblib</b> to enable inference.</div>', unsafe_allow_html=True)
else:
    st.success("Model online and ready for inference")

st.subheader("Upload traffic records")
upload = st.file_uploader("CSV input", type=["csv"], label_visibility="collapsed")
if upload:
    frame = pd.read_csv(upload)
    a, b, c = st.columns(3)
    a.metric("Rows detected", f"{len(frame):,}")
    b.metric("Input features", f"{len(frame.columns):,}")
    c.metric("Schema status", "Ready" if not health.get("required_columns") else "Validated")
    with st.expander("Preview input", expanded=False):
        st.dataframe(frame.head(20), use_container_width=True)
    if st.button("Run batch detection", type="primary", use_container_width=True):
        try:
            response = requests.post(f"{API_URL}/api/predict", files={"file": (upload.name, upload.getvalue(), "text/csv")}, timeout=120)
            data = response.json()
            if response.status_code != 200:
                st.error(data.get("error", "Prediction unavailable"))
            else:
                st.session_state["result"] = data
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Detection summary")
    cols = st.columns(max(1, len(result["class_distribution"])))
    for column, (label, count) in zip(cols, result["class_distribution"].items()):
        column.metric(f"{label}", f"{count:,}")
    st.dataframe(pd.DataFrame(result["predictions"]), use_container_width=True)
    csv = pd.DataFrame(result["predictions"]).to_csv(index=False).encode()
    st.download_button("Download predictions", csv, "ids_predictions.csv", "text/csv", use_container_width=True)

with st.expander("Explainability and monitoring"):
    st.caption("SHAP feature-importance is active below. Confusion-matrix, ROC, and drift panels still need ground-truth labels / a reference baseline and are not wired yet.")

    if not upload:
        st.info("Upload a CSV above to generate a SHAP explanation report.")
    elif health["status"] != "ready":
        st.info("Model artifact not connected. Add models/random_forest_pipeline4.joblib to enable explanations.")
    else:
        if st.button("Generate SHAP explanation report", use_container_width=True):
            try:
                response = requests.post(
                    f"{API_URL}/api/explain",
                    files={"file": (upload.name, upload.getvalue(), "text/csv")},
                    timeout=180,
                )
                data = response.json()
                if response.status_code != 200:
                    st.error(data.get("error", "Explanation unavailable"))
                else:
                    st.session_state["explain_result"] = data
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")

        explain_result = st.session_state.get("explain_result")
        if explain_result:
            st.divider()
            st.markdown(
                f"**Explained class:** `{explain_result.get('explained_class')}` &nbsp;·&nbsp; "
                f"**Rows explained:** {explain_result.get('rows_explained', 0):,}"
            )

            png_b64 = explain_result.get("summary_plot_png_base64")
            if png_b64:
                st.image(base64.b64decode(png_b64), caption="SHAP feature importance", use_container_width=True)

            importance = explain_result.get("feature_importance", [])
            if importance:
                st.subheader("Top features (mean |SHAP value|)")
                importance_df = pd.DataFrame(importance[:15]).set_index("feature")
                st.bar_chart(importance_df["mean_abs_shap"])

            row_explanations = explain_result.get("row_explanations", [])
            if row_explanations:
                st.subheader("Per-row feature contributions")
                row_choice = st.selectbox(
                    "Row",
                    options=[item["row"] for item in row_explanations],
                    format_func=lambda i: f"Row {i}",
                )
                selected = next(item for item in row_explanations if item["row"] == row_choice)
                st.dataframe(pd.DataFrame(selected["top_features"]), use_container_width=True, hide_index=True)
