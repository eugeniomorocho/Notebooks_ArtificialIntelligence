import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import io

# Advanced Streamlit tutorial:
st.title("Streamlit — Advanced: Interactivity, Caching & Deployment")
st.markdown(
    "This app demonstrates advanced Streamlit patterns: session state, caching, "
    "file upload, dynamic forms, model training, prediction and downloadable results."
)

# ---------------------------------------------------------------------
# Helper: load a small sample dataset for quick experiments (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_sample():
    from sklearn.datasets import load_breast_cancer
    b = load_breast_cancer(as_frame=True)
    df = b.frame
    df.rename(columns={"target": "TARGET"}, inplace=True)
    return df

# ---------------------------------------------------------------------
# Data input: upload CSV or use sample
# ---------------------------------------------------------------------
st.sidebar.header("Data input")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_sample = st.sidebar.button("Use sample dataset") if uploaded is None else False

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        st.stop()
elif use_sample or uploaded is None:
    df = load_sample()

st.subheader("Dataset preview")
st.write("Rows, cols:", df.shape)
st.dataframe(df.head())

# ---------------------------------------------------------------------
# Dynamic selection of features and target
# ---------------------------------------------------------------------
st.sidebar.header("Model setup")
all_cols = df.columns.tolist()
target_col = st.sidebar.selectbox("Target column", options=all_cols, index=len(all_cols)-1)
feature_cols = st.sidebar.multiselect("Feature columns (choose some)", options=[c for c in all_cols if c != target_col],
                                      default=[c for c in all_cols if c != target_col][:6])

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

# Simple EDA
st.subheader("Simple EDA")
st.write("Target distribution:")
st.bar_chart(df[target_col].value_counts(normalize=True))

# ---------------------------------------------------------------------
# Train / persist model using session state
# ---------------------------------------------------------------------
if "model_pipeline" not in st.session_state:
    st.session_state.model_pipeline = None
    st.session_state.feature_names = None

train_col1, train_col2 = st.columns([2, 1])
with train_col1:
    test_size_pct = st.slider("Test size (%)", 10, 40, 20)
    random_state = int(st.number_input("Random seed", value=42, step=1))
    n_estimators = int(st.number_input("RF n_estimators", value=100, step=10))
with train_col2:
    run_train = st.button("Train model")
    download_btn_placeholder = st.empty()

# Prepare data
X = df[feature_cols]
y = df[target_col]

# Auto-detect numerical and categorical features
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_features),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_features),
    ],
    remainder="drop",
)

# Train
if run_train:
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_pct/100.0, stratify=y, random_state=random_state
        )
        model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
        model.fit(X_train, y_train)

        # Save to session state
        st.session_state.model_pipeline = model
        st.session_state.feature_names = feature_cols

    st.success("Model trained and stored in session state ✅")

    # Evaluate
    y_pred = model.predict(X_test)
    st.subheader("Evaluation")
    st.text(classification_report(y_test, y_pred, digits=4))
    if hasattr(model.named_steps["randomforestclassifier"], "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            st.write("ROC AUC:", f"{roc_auc_score(y_test, y_proba):.4f}")
        except Exception:
            pass

    # Offer download of the trained pipeline
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    download_btn_placeholder.download_button("Download trained pipeline (joblib)", data=buf, file_name="pipeline.joblib")

# ---------------------------------------------------------------------
# If a model exists, allow predictions on new data
# ---------------------------------------------------------------------
if st.session_state.model_pipeline is not None:
    st.subheader("Predict on dataset / upload new CSV")
    predict_choice = st.radio("Use:", ["Current dataset", "Upload new CSV"], index=0)
    if predict_choice == "Current dataset":
        pred_df = df.copy()
    else:
        up = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_csv")
        if up is not None:
            try:
                pred_df = pd.read_csv(up)
            except Exception as e:
                st.error(f"Cannot read CSV: {e}")
                st.stop()
        else:
            st.info("Upload a CSV to run predictions.")
            pred_df = None

    if pred_df is not None:
        # Ensure required feature columns exist
        missing = [c for c in st.session_state.feature_names if c not in pred_df.columns]
        if missing:
            st.error(f"Missing columns for prediction: {missing}")
        else:
            preds = st.session_state.model_pipeline.predict(pred_df[st.session_state.feature_names])
            pred_df["_prediction"] = preds
            st.dataframe(pred_df.head())
            # Download predictions
            out_buf = io.BytesIO()
            pred_df.to_csv(out_buf, index=False)
            out_buf.seek(0)
            st.download_button("Download predictions CSV", data=out_buf, file_name="predictions.csv")

# ---------------------------------------------------------------------
# Optional: SHAP explanations (if installed)
# ---------------------------------------------------------------------
st.subheader("Model explainability (optional)")
try:
    import shap
    if st.session_state.model_pipeline is None:
        st.info("Train a model to enable SHAP explanations.")
    else:
        # Compute SHAP for a small sample to avoid long compute
        model = st.session_state.model_pipeline
        # Try to extract the fitted tree estimator
        try:
            rf = model.named_steps["randomforestclassifier"]
            # Preprocess a small X sample
            X_sample = X.head(100)
            X_trans = model.named_steps["columntransformer"].transform(X_sample)
            expl = shap.TreeExplainer(rf)
            shap_values = expl.shap_values(X_trans)
            st.write("SHAP computed (sample). Use a notebook for interactive plots.")
        except Exception as e:
            st.warning("SHAP explanation not available for this pipeline: " + str(e))
except Exception:
    st.info("Install `shap` to enable model explainability (optional).")

# ---------------------------------------------------------------------
# Helpful notes for students
# ---------------------------------------------------------------------
st.markdown(
    """
    **Notes for students**
    - Use the sidebar to select features and target; experiment with different sample sizes and model parameters.
    - Session state keeps the trained model during the app lifetime — useful for iterative exploration.
    - To deploy: commit the file and run `streamlit run 3_streamlit_advanced.py` on a server
      or use Streamlit Cloud / Streamlit Community deployment.
    """
)