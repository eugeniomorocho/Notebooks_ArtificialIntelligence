# ...existing code...
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib

# App header
st.title("Streamlit — Data Scientist Mini App (stable)")

# Load a public dataset (cached)
@st.cache_data
def load_data():
    b = load_breast_cancer(as_frame=True)
    X = b.frame.drop(columns=["target"])
    y = b.frame["target"]
    return X, y, b.feature_names

X, y, feature_names = load_data()

# Sidebar controls (ensure these are defined before training)
st.sidebar.header("Experiment settings")
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])
n_estimators = int(st.sidebar.slider("RF n_estimators", 50, 500, 100)) if model_choice == "Random Forest" else None
sample_frac = st.sidebar.slider("Sample fraction (for quick runs)", 0.1, 1.0, 1.0, step=0.1)

# Basic dataset preview and class balance
st.subheader("Dataset preview")
if st.checkbox("Show raw data (first 10 rows)"):
    st.dataframe(pd.concat([X, y], axis=1).head(10))

st.write("Features:", len(feature_names))
st.write("Class balance (positive = malignant):")
st.bar_chart(y.value_counts(normalize=True))

# Subsample for speed if requested
if sample_frac < 1.0:
    df = pd.concat([X, y], axis=1).sample(frac=sample_frac, random_state=random_state)
    X_run = df.drop(columns=["target"])
    y_run = df["target"]
else:
    X_run, y_run = X, y

# Train/test split (defined before the Train button)
X_train, X_test, y_train, y_test = train_test_split(
    X_run, y_run, test_size=test_size/100.0, stratify=y_run, random_state=random_state
)

# Train button
if st.button("Train model"):
    with st.spinner("Training model — this may take a few seconds..."):
        # Choose classifier
        if model_choice == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

        # Build pipeline and fit
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            clf
        )
        pipeline.fit(X_train, y_train)

    st.success("Training finished ✅")

    # Predictions and metrics
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    st.subheader("Evaluation metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    if not np.isnan(auc):
        st.write(f"ROC AUC: {auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.subheader("Confusion matrix")
    st.pyplot(fig)

    # Feature importance / coefficients
    st.subheader("Feature importance / coefficients")
    clf_name = clf.__class__.__name__.lower()
    try:
        trained_clf = pipeline.named_steps[clf_name]
    except Exception:
        # Fallback: classifier is last step
        trained_clf = pipeline.steps[-1][1]

    if isinstance(trained_clf, LogisticRegression):
        coefs = trained_clf.coef_.ravel()
        importance = pd.Series(coefs, index=feature_names).abs().sort_values(ascending=False)
    else:
        importances = trained_clf.feature_importances_
        importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    st.table(importance.head(10))

    # Save model and offer download
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)
    st.download_button("Download trained pipeline (joblib)", data=buf, file_name="pipeline.joblib")
else:
    st.info("Adjust settings in the sidebar and press 'Train model' to run a quick experiment.")