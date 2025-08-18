# Sarcasm-and-Toxicity-detection-in-Reddt-Comments-Using-GloVe
The project aimed at predicting  sarcasm and  toxicity based on Kaggle Sarcasm Headlines dataset using GloVe embeddings, Logistic Regression and Decision Tree Classifier.
# Objectives
The main objectives of the project are:
##### To preprocess and vectorize text data using pre-trained GloVe embeddings.
##### To train and evaluate Logistic Regression and Decision Tree classifiers for sarcasm detection.
##### To compare performance metrics such as Precision, Recall, F1-score, and ROC-AUC.
##### To implement an interactive prediction tool that allows users to analyze new input.
##### To provide a simple toxicity heuristic by scanning input text against a curated toxic word lexicon.

# Codes 
import io
import json
import gzip
import re
from typing import Dict, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

## STREAMLIT PAGE CONFIG
st.set_page_config(page_title="Sarcasm & Toxicity Detector", page_icon="🛰️", layout="wide")
st.title("🛰️ Sarcasm & Toxicity Detector")
st.markdown(
    """
 **Task:** Sarcasm classification + toxicity analysis  
 **Embeddings:** GloVe (averaged word vectors) — supports `.txt` and `.txt.gz`  
 **Models:** Logistic Regression & Decision Tree  
 **Metrics:** Precision, Recall, F1, ROC-AUC  
 **Dataset:** Kaggle Sarcasm Headline Dataset (JSON/JSONL)
    """
)

 ---- Session State Initialization
needed_keys = {
    "glove_vectors": None, "glove_dim": None,
    "model_lr": None, "model_dt": None,
    "metrics_lr": None, "metrics_dt": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "y_pred_lr": None, "y_pred_dt": None,
    "y_prob_lr": None, "y_prob_dt": None,
}
for k, v in needed_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

#### Removal of stop words

_url_re = re.compile(r"http[s]?://\S+|www\.\S+", flags=re.IGNORECASE)
_non_alnum_re = re.compile(r"[^a-z0-9'\s]+")


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = _url_re.sub(" ", s)
    s = _non_alnum_re.sub(" ", s)
    return " ".join(s.split())


@st.cache_data(show_spinner=False)
def load_dataset(uploaded: Any) -> pd.DataFrame:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore").strip()
    if raw.startswith("["):
        data = json.loads(raw)
        df = pd.DataFrame(data)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        df = pd.DataFrame(rows)

    if not {"headline", "is_sarcastic"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'headline' and 'is_sarcastic'.")

    out = df[["headline", "is_sarcastic"]].rename(columns={"headline": "text", "is_sarcastic": "label"})
    out["text"] = out["text"].astype(str).apply(clean_text)
    out["label"] = out["label"].astype(int)
    return out


@st.cache_data(show_spinner=True)
def load_glove_bytes(content: bytes, filename: str) -> Dict[str, np.ndarray]:
    if filename.lower().endswith(".gz"):
        try:
            text = gzip.decompress(content).decode("utf-8", errors="ignore")
        except Exception:
            with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as gf:
                text = gf.read().decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    vocab: Dict[str, np.ndarray] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0]
        try:
            vec = np.asarray(parts[1:], dtype=np.float32)
        except ValueError:
            continue
        if vec.size > 0:
            vocab[word] = vec
    return vocab


def infer_glove_dim(embeds: Dict[str, np.ndarray]) -> int:
    if not embeds:
        return 0
    first = next(iter(embeds.values()))
    return int(first.shape[0])


def sentence_vec(text: str, glove: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    tokens = text.split()
    vecs = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


TOXIC_LEXICON = {
    "stupid", "idiot", "dumb", "trash", "hate", "ugly", "moron", "loser", "fuck", "empty mind",
    "nonsense", "nazi", "racist", "shut", "kill", "terrible", "worthless", "bad guy", "silly",
    "garbage", "suck", "crazy", "disgusting", "fool", "pathetic", "toxic", "mad", "hurt","Nonsense",
    "useless", "Failure", "Loser", "Weak", "Pathetic", "Fake", "Awful", "Terrible", "Incompetent", "Manipulative"}

def toxicity_score(text: str) -> float:
    toks = clean_text(text).split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in TOXIC_LEXICON)
    return hits / max(3, len(toks))
    
st.sidebar.header("Upload files")
dataset_file = st.sidebar.file_uploader("Kaggle Sarcasm JSON/JSONL", type=["json", "jsonl"])

glove_file = st.sidebar.file_uploader(
    "Upload GloVe (.txt or .txt.gz) — e.g., glove.6B.100d.txt or .txt.gz",
    type=["txt", "gz"],
    key="glove",
)

seed = st.sidebar.number_input("Random seed", value=42, step=1)
do_train = st.sidebar.button("Train models", type="primary")

st.subheader("Feature Selection")
if st.checkbox('Show the clean data and Check for class imbalance'):
    if dataset_file:
        df = load_dataset(dataset_file)
        st.success(f"Loaded {len(df):,} headlines")

        with st.expander("Preview dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        counts = df["label"].value_counts().rename({0: "Not Sarcastic", 1: "Sarcastic"})
        fig = plt.figure()
        counts.plot(kind="bar")
        plt.ylabel("Count")
        plt.title("Class distribution")
        st.pyplot(fig)
    else:
        st.info("Upload the dataset to continue.")
        
## Train and evaluate the performance of the model
st.subheader("Training and Evaluation of the model")
if st.checkbox('Model Training and Evaluation'):
    if st.checkbox('Classification Report'):
        if do_train:
            if not dataset_file or not glove_file:
                st.error("Please upload both the dataset and a GloVe file (.txt or .txt.gz).")
                st.stop()

            df = load_dataset(dataset_file)
            glove_bytes = glove_file.getvalue()
            glove_vectors = load_glove_bytes(glove_bytes, getattr(glove_file, "name", "glove.txt"))
            dim = infer_glove_dim(glove_vectors)
            if dim == 0:
                st.error("Could not infer GloVe dimension. Check the file.")
                st.stop()
            st.info(f"GloVe dimension detected: {dim}")

            X = np.vstack([sentence_vec(t, glove_vectors, dim) for t in df["text"].tolist()])
            y = df["label"].values.astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=int(seed), stratify=y
            )

            model_lr = LogisticRegression(max_iter=300, solver="liblinear", random_state=int(seed))
            model_lr.fit(X_train, y_train)
            y_pred_lr = model_lr.predict(X_test)
            y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

            model_dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, random_state=int(seed))
            model_dt.fit(X_train, y_train)
            y_pred_dt = model_dt.predict(X_test)
            y_prob_dt = model_dt.predict_proba(X_test)[:, 1] if hasattr(model_dt,
                                                                        "predict_proba") else y_pred_dt.astype(float)


            def summarize(y_true, y_pred, y_score):
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                try:
                    auc = roc_auc_score(y_true, y_score)
                except Exception:
                    auc = float("nan")
                return {"precision": p, "recall": r, "f1": f1, "roc_auc": auc}


            metrics_lr = summarize(y_test, y_pred_lr, y_prob_lr)
            metrics_dt = summarize(y_test, y_pred_dt, y_prob_dt)

            st.session_state.update({
                "glove_vectors": glove_vectors,
                "glove_dim": dim,
                "model_lr": model_lr,
                "model_dt": model_dt,
                "metrics_lr": metrics_lr,
                "metrics_dt": metrics_dt,
                "X_train": X_train,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred_lr": y_pred_lr,
                "y_pred_dt": y_pred_dt,
                "y_prob_lr": y_prob_lr,
                "y_prob_dt": y_prob_dt
            })

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Logistic Regression — Metrics")
                st.json({k: round(v, 3) for k, v in metrics_lr.items()})
                st.code(classification_report(y_test, y_pred_lr, digits=3), language="text")
            with col2:
                st.subheader("Decision Tree — Metrics")
                st.json({k: round(v, 3) for k, v in metrics_dt.items()})
                st.code(classification_report(y_test, y_pred_dt, digits=3), language="text")
    if st.checkbox('ROC Curve and AU-ROC'):
        ready_for_roc = (
                st.session_state["y_test"] is not None and
                st.session_state["y_prob_lr"] is not None and
                st.session_state["y_prob_dt"] is not None and
                st.session_state["metrics_lr"] is not None and
                st.session_state["metrics_dt"] is not None
        )

        if not ready_for_roc:
            st.info("Train the models first (click **Train models**) to see ROC and confusion matrices.")
        else:
            y_test = st.session_state["y_test"]
            y_prob_lr = st.session_state["y_prob_lr"]
            y_prob_dt = st.session_state["y_prob_dt"]

            fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
            fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

            # Single, clean ROC figure
            fig = plt.figure()
            plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={st.session_state['metrics_lr']['roc_auc']:.3f})")
            plt.plot(fpr_dt, tpr_dt, label=f"DecisionTree (AUC={st.session_state['metrics_dt']['roc_auc']:.3f})")
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves")
            plt.legend(loc="lower right")
            st.pyplot(fig)

            # Optional confusion matrices
            if st.checkbox("CM for Logistic Regression"):
                fig_cm1, ax1 = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    y_test, st.session_state["y_pred_lr"], ax=ax1, colorbar=False
                )
                ax1.set_title("LogReg — Confusion Matrix")
                st.pyplot(fig_cm1)

            if st.checkbox("CM for Decision Tree"):
                fig_cm2, ax2 = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    y_test, st.session_state["y_pred_dt"], ax=ax2, colorbar=False
                )
                ax2.set_title("Decision Tree — Confusion Matrix")
                st.pyplot(fig_cm2)

             ------------------------ Hyperparameter tuning (optional) -----------------
            if st.checkbox("Hyperparameter tuning"):
                # Safe keys for training data
                need_keys_hp = ["X_train", "y_train"]
                if not all(k in st.session_state for k in need_keys_hp):
                    st.warning("Please run training first to populate X_train/y_train.")
                else:
                    X_train = st.session_state["X_train"]
                    y_train = st.session_state["y_train"]

                    # Pipeline with a single step "clf" so param grid uses "clf__*"
                    pipe = Pipeline([("clf", LogisticRegression())])

                    # Two model spaces: LR and DT, using the same 'clf' step
                    param_grid = [
                        {
                            "clf": [LogisticRegression(max_iter=500, random_state=int(seed))],
                            "clf__C": [0.1, 1.0, 10.0],
                            "clf__penalty": ["l2"],
                            "clf__solver": ["lbfgs", "liblinear"],
                        },
                        {
                            "clf": [DecisionTreeClassifier(random_state=int(seed))],
                            "clf__max_depth": [10, 20, None],
                            "clf__min_samples_split": [2, 5, 10],
                            "clf__criterion": ["gini", "entropy"],
                        },
                    ]

                    grid = GridSearchCV(
                        estimator=pipe,
                        param_grid=param_grid,
                        cv=5,
                        scoring="f1_weighted",   # works with binary and slight imbalance
                        n_jobs=-1,
                        verbose=1,
                    )
                    grid.fit(X_train, y_train)

                    st.write("**Best Params:**", grid.best_params_)
                    st.success(f"Accuracy of the best model: {grid.best_score_:.4f}")

                    # Persist best tuned model for predictions
                    best_clf = grid.best_estimator_.named_steps["clf"]
                    st.session_state["best_model"] = best_clf
                    st.session_state["best_model_name"] = type(best_clf).__name__

#### Toxicity prediction based on the best model

st.subheader("Make Prediction")

if st.checkbox('Make prediction on unseen data'):
    # Check training artifacts exist
    base_ready = all(k in st.session_state for k in ["glove_vectors", "glove_dim"])
    models_ready = all(k in st.session_state for k in ["model_lr", "model_dt", "metrics_lr", "metrics_dt"])
    if not (base_ready and models_ready):
        st.info("Train the models first to enable live predictions.")
    else:
        user_input = st.text_area("Enter text for analysis", height=120, placeholder="Type a headline or message...")

        colA, colB = st.columns(2)
        with colA:
            run_pred = st.button("Predict")
        with colB:
            uploaded_batch = st.file_uploader(
                "Or upload a file for batch prediction (.txt, .csv or .json)",
                type=["txt", "csv", "json"],
                key="batch_file"
            )

        # Decide which model to use:
        # 1) Prefer tuned best model if available
        # 2) else pick the baseline with the higher ROC-AUC
        def pick_best_model():
            if "best_model" in st.session_state:
                return st.session_state["best_model"], st.session_state.get("best_model_name", "TunedBest")
            # fallback: compare baseline ROC-AUCs
            auc_lr = st.session_state["metrics_lr"].get("roc_auc", float("nan"))
            auc_dt = st.session_state["metrics_dt"].get("roc_auc", float("nan"))
            if (not np.isnan(auc_lr)) and (auc_lr >= (auc_dt if not np.isnan(auc_dt) else -1)):
                return st.session_state["model_lr"], "LogisticRegression"
            else:
                return st.session_state["model_dt"], "DecisionTree"

        best_model, best_name = pick_best_model()

        def predict_one(text: str):
            text_clean = clean_text(text)
            vec = sentence_vec(text_clean, st.session_state["glove_vectors"], st.session_state["glove_dim"]).reshape(1, -1)
            pred = int(best_model.predict(vec)[0])
            proba = float(best_model.predict_proba(vec)[0, 1]) if hasattr(best_model, "predict_proba") else None
            tox = toxicity_score(text)  # your heuristic toxicity score (leave as-is)
            return pred, proba, tox

        if run_pred:
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                pred, proba, tox = predict_one(user_input)
                label = "Sarcastic" if pred == 1 else "Not Sarcastic"
                st.markdown(f"• **Model used:** `{best_name}`")
                st.markdown(f"• Predicted class: **{label}**")
                if proba is not None:
                    st.markdown(f"• P(class = Sarcastic): **{proba:.3f}**")
                st.markdown(f"• Toxicity (lexicon score): **{tox:.3f}**  (demo heuristic)")

        if uploaded_batch is not None:
            try:
                fname = uploaded_batch.name.lower()
                if fname.endswith(".txt"):
                    lines = uploaded_batch.read().decode("utf-8", errors="ignore").splitlines()
                    batch_df = pd.DataFrame({"text": [l for l in lines if l.strip()]})
                elif fname.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_batch)
                else:  # .json (array or JSON Lines)
                    try:
                        batch_df = pd.read_json(uploaded_batch, lines=True)
                    except ValueError:
                        uploaded_batch.seek(0)
                        import json as _json
                        data = _json.load(uploaded_batch)
                        batch_df = pd.DataFrame(data)

                # normalize column name to "text"
                candidate_cols = [c for c in batch_df.columns if c.lower() in {"text", "headline", "message", "content"}]
                if candidate_cols:
                    first = candidate_cols[0]
                    if first != "text":
                        batch_df = batch_df.rename(columns={first: "text"})
                elif "text" not in batch_df.columns:
                    raise ValueError("Input must contain a 'text' column (or headline/message/content).")

                preds, probas, toxes = [], [], []
                for t in batch_df["text"].astype(str).tolist():
                    p, pr, tx = predict_one(t)
                    preds.append("Sarcastic" if p == 1 else "Not Sarcastic")
                    probas.append(pr if pr is not None else np.nan)
                    toxes.append(tx)

                batch_df["Predicted Class"] = preds
                batch_df["Prob(Sarcastic)"] = probas
                batch_df["ToxicityScore"] = toxes

                st.success(f"Batch processed successfully with `{best_name}`!")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")

