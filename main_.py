# main_with_session.py
"""
Merged version of your app with:
- robust label mapping & preprocessing (from your main_.py)
- SHAP caching fix (underscore arg) + uncached fallback
- session_state persistence so selectbox changes don't reset everything
Save and run:
    streamlit cache clear
    streamlit run main_with_session.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import io
from typing import Tuple

# Page config
st.set_page_config(page_title="IDS + XAI (Persistent)", layout="wide")

# --------------------------
# Helper functions (from your code)
# --------------------------
@st.cache_data(show_spinner=False)
def load_dataset_from_file(file_bytes: bytes, sep: str = ',') -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine='python')
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=r'\s+', engine='python', header=None)
    return df

def default_load(path='KDDTrain+.txt') -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Default dataset not found at {path}. Please upload a file in the UI.")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    return df

def assign_kdd_columns(df: pd.DataFrame) -> pd.DataFrame:
    kdd_cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    if df.shape[1] == len(kdd_cols) + 1:
        df.columns = kdd_cols + ['target']
    elif df.shape[1] == len(kdd_cols):
        df.columns = kdd_cols
    return df

def robust_label_mapping(df: pd.DataFrame) -> pd.Series:
    if 'target' not in df.columns:
        raise ValueError("Dataset must contain a 'target' column (attack label).")

    raw = df['target'].astype(str).str.lower().str.strip()
    unique_vals = list(pd.Series(raw.unique()).astype(str))

    # Numeric-like 0/1
    if set(unique_vals) <= {'0','1'}:
        return pd.to_numeric(raw, errors='coerce').fillna(0).astype(int)

    # If 'normal' present
    if any('normal' == v for v in unique_vals):
        return raw.apply(lambda x: 0 if x == 'normal' else 1)

    # benign/background
    if any('benign' == v for v in unique_vals):
        return raw.apply(lambda x: 0 if x == 'benign' else 1)
    if any('background' == v for v in unique_vals):
        return raw.apply(lambda x: 0 if x == 'background' else 1)

    # fallback: most frequent class -> normal
    counts = raw.value_counts()
    most_freq = counts.idxmax()
    return raw.apply(lambda x: 0 if x == most_freq else 1)

def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if 'target' not in df.columns and df.shape[1] > 1:
        cols = list(df.columns)
        cols[-1] = 'target'
        df.columns = cols

    if 'target' not in df.columns or df.shape[1] < 5:
        df = assign_kdd_columns(df)

    if 'target' not in df.columns:
        raise ValueError("Dataset must contain a 'target' column (attack label).")

    df['label'] = robust_label_mapping(df)

    # encode categorical columns
    cat_cols = [
        c for c in df.columns
        if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        and c != 'target'
    ]
    le = LabelEncoder()
    for c in cat_cols:
        try:
            df[c] = le.fit_transform(df[c].astype(str))
        except Exception:
            df[c] = df[c].astype('category').cat.codes

    drop_cols = []
    if 'target' in df.columns:
        drop_cols.append('target')

    X = df.drop(columns=drop_cols + ['label'], errors='ignore')
    y = df['label'].astype(int)

    return X, y

@st.cache_resource(show_spinner=False)
def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# SHAP explainer: cached safe version (underscore param) + uncached fallback
@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model, background_data: pd.DataFrame, nsample: int = 200):
    import shap
    bg = background_data.sample(n=min(nsample, len(background_data)), random_state=42)
    explainer = shap.TreeExplainer(_model, data=bg, model_output='probability')
    return explainer

def get_shap_explainer_uncached(model, background_data: pd.DataFrame, nsample: int = 200):
    import shap
    bg = background_data.sample(n=min(nsample, len(background_data)), random_state=42)
    explainer = shap.TreeExplainer(model, data=bg, model_output='probability')
    return explainer

# --------------------------
# UI: controls and session persistence
# --------------------------
st.title("Anomaly IDS")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload dataset", type=['csv','txt','data'])
    sep_choice = st.selectbox("Delimiter", options=[',','\\s+'], index=0)
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20)
    n_estimators = st.slider("RandomForest n_estimators", 10, 500, 100)
    run_button = st.button("Train Model & Run Detection")
    clear_session = st.button("Clear saved model & explainer")

# Clear session if user requests
if clear_session:
    keys = ['model','X_test_reset','y_test_reset','preds','probs','flagged_indices','explainer','selected_attack','accuracy','conf_matrix','X_train','y_train']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Session state cleared. Re-run training when ready.")

# If a new file is uploaded clear prior model/explainer to avoid mismatch
if uploaded_file is not None:
    keys_to_clear = ['model','X_test_reset','y_test_reset','preds','probs','flagged_indices','explainer','selected_attack','accuracy','conf_matrix','X_train','y_train']
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

# Load dataset
df = None
if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
        sep = ',' if sep_choice == ',' else r'\s+'
        df = load_dataset_from_file(raw_bytes, sep=sep)
        st.success("Uploaded dataset loaded.")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
else:
    try:
        df = default_load('KDDTrain+.txt')
        st.info("Loaded default dataset from KDDTrain+.txt")
    except Exception:
        df = None

if df is not None:
    st.write("Preview (first 5 rows):")
    st.dataframe(df.head(5))

# Decide whether to (re)train: if user clicked run OR no model exists in session_state
need_train = run_button

# Training / prepare model and session_state objects
if need_train:
    if df is None:
        st.error("No dataset available to train. Upload a dataset or place KDDTrain+.txt in working directory.")
        st.stop()

    try:
        X, y = preprocess_df(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # Show label counts for debug
    st.write("Label distribution after mapping:")
    st.write(y.value_counts().to_frame("count"))
    if y.nunique() < 2:
        st.error("After mapping, there is only a single class in labels — training cannot proceed. Check your 'target' column.")
        st.stop()

    # Split
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=stratify)

    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    with st.spinner("Training RandomForest..."):
        model = train_rf_model(X_train, y_train, n_estimators=n_estimators)

    preds = model.predict(X_test)
    proba_matrix = model.predict_proba(X_test)
    if proba_matrix.ndim == 1 or proba_matrix.shape[1] == 1:
        probs = proba_matrix.reshape(-1)
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-12)
    else:
        probs = proba_matrix[:, 1]

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # Persist items in session_state
    st.session_state.model = model
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test_reset = X_test_reset
    st.session_state.y_test_reset = y_test_reset
    st.session_state.preds = preds
    st.session_state.probs = probs
    st.session_state.flagged_indices = [int(i) for i, p in enumerate(preds) if p == 1]
    st.session_state.accuracy = acc
    st.session_state.conf_matrix = cm

# If model exists in session_state, use it (persistent)
if 'model' in st.session_state:
    model = st.session_state.model
    X_test_reset = st.session_state.X_test_reset
    y_test_reset = st.session_state.y_test_reset
    preds = st.session_state.preds
    probs = st.session_state.probs
    flagged_indices = st.session_state.flagged_indices
    acc = st.session_state.accuracy
    cm = st.session_state.conf_matrix

    st.success(f"Model ready. Test accuracy: {acc:.4f}")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.write("### Classification Report")
        st.text(classification_report(y_test_reset, preds))
        st.write(f"**Accuracy:** {acc:.4f}")
    with col2:
        st.write("### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"], cbar=False, ax=ax_cm)
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm); plt.close(fig_cm)

    st.write(f"Total attacks flagged (test set): {len(flagged_indices)}")
    if len(flagged_indices) > 0:
        preview = X_test_reset.iloc[flagged_indices[:10]].copy()
        preview['pred_prob'] = np.array(probs)[flagged_indices[:10]]
        st.write("First flagged samples (preview):")
        st.dataframe(preview)

    # Prepare SHAP explainer once and store in session_state
    if 'explainer' not in st.session_state:
        with st.spinner("Initializing SHAP explainer (cached if possible)..."):
            explainer = None
            try:
                # cached safe explainer (note underscore param)
                explainer = get_shap_explainer(model, st.session_state.X_train, nsample=200)
                st.session_state.explainer = explainer
                st.success("SHAP explainer cached.")
            except Exception as e_cached:
                st.warning(f"Cached explainer init failed: {e_cached}")
                try:
                    explainer = get_shap_explainer_uncached(model, st.session_state.X_train, nsample=100)
                    st.session_state.explainer = explainer
                    st.warning("Using uncached explainer (stored in session).")
                except Exception as e_uncached:
                    st.error(f"Explainer initialization failed: {e_uncached}")
                    st.session_state.explainer = None

    explainer = st.session_state.get('explainer', None)
    shap_module = None
    if explainer is not None:
        try:
            import shap as shap_module
        except Exception as e:
            st.error(f"SHAP could not be loaded: {e}")

    # SHAP global summary
    st.subheader("🧠 SHAP Explainability (Global Feature Impact)")
    if explainer is None or shap_module is None:
        st.warning("SHAP explainer not available.")
    else:
        if len(flagged_indices) > 0:
            idxs = flagged_indices[:200]
            sample_data = X_test_reset.iloc[idxs]
        else:
            sample_data = X_test_reset.iloc[:200]

        try:
            shap_values_obj = explainer(sample_data)
            if hasattr(shap_values_obj, "values") and shap_values_obj.values.ndim == 3:
                shap_vals_class1 = shap_values_obj.values[:, :, 1]
            else:
                shap_vals_class1 = shap_values_obj.values

            abs_mean = np.mean(np.abs(shap_vals_class1), axis=0)
            top_k = min(20, len(abs_mean))
            top_idx = np.argsort(abs_mean)[-top_k:][::-1]
            top_features = list(sample_data.columns[top_idx])
            sample_subset = sample_data[top_features]
            shap_subset = shap_vals_class1[:, top_idx]

            fig_shap = plt.figure(figsize=(9, 6))
            shap_module.summary_plot(shap_subset, sample_subset, feature_names=top_features, plot_type="dot", show=False, max_display=top_k)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except Exception as e:
            st.error(f"Failed to compute SHAP global summary: {e}")

    # Local SHAP (per-sample) with session-bound selectbox
    st.subheader("🔍 Local SHAP Explanation (per-sample)")
    if 'selected_attack' not in st.session_state:
        st.session_state.selected_attack = flagged_indices[0] if len(flagged_indices) > 0 else 0

    if len(flagged_indices) > 0:
        # keep selectbox bound to session_state selected_attack
        idx_default_pos = flagged_indices.index(st.session_state.selected_attack) if st.session_state.selected_attack in flagged_indices else 0
        selection = st.selectbox("Select flagged attack (positional index in test set):", flagged_indices, index=idx_default_pos, key='selected_attack')
        sample_idx = int(selection)
    else:
        sample_idx = st.number_input("Choose test sample index:", min_value=0, max_value=len(X_test_reset)-1, value=st.session_state.selected_attack if 'selected_attack' in st.session_state else 0, step=1)
        st.session_state.selected_attack = int(sample_idx)

    sample = X_test_reset.iloc[[sample_idx]]
    st.write("Selected sample preview:")
    st.dataframe(sample.T)

    instance_shap = None
    if explainer is not None:
        with st.spinner("Computing local SHAP..."):
            try:
                instance_shap = explainer(sample)
            except Exception as e:
                st.error(f"Local SHAP failed: {e}")
                instance_shap = None

    pred_class = int(model.predict(sample)[0])
    try:
        pm = model.predict_proba(sample)
        if pm.ndim == 1 or pm.shape[1] == 1:
            proba_for_display = float(pm.reshape(-1)[0])
        else:
            proba_for_display = float(pm[0, 1])
    except Exception:
        proba_for_display = float(probs[sample_idx] if sample_idx < len(probs) else 0.0)

    risk = "HIGH" if proba_for_display > 0.85 else "MEDIUM" if proba_for_display > 0.5 else "LOW"
    st.metric("Prediction", "Attack" if pred_class == 1 else "Normal")
    st.metric("Attack Probability", f"{proba_for_display:.4f}")
    st.metric("Risk Level", risk)

    if instance_shap is not None:
        try:
            if instance_shap.values.ndim == 3:
                shap_vals_inst = instance_shap.values[0][:, pred_class]
            else:
                shap_vals_inst = instance_shap.values[0]

            shap_df = pd.DataFrame({
                "Feature": sample.columns,
                "Value": sample.values[0],
                "SHAP": shap_vals_inst
            })
            shap_df['Abs'] = shap_df['SHAP'].abs()
            shap_df = shap_df.sort_values("Abs", ascending=False).reset_index(drop=True)
            st.write("Top contributing features (by absolute SHAP):")
            st.dataframe(shap_df[['Feature','Value','SHAP']].head(10))

            top_k_local = min(12, len(shap_df))
            try:
                fig_wf = plt.figure(figsize=(9,5))
                shap_module.plots.waterfall(instance_shap[0], max_display=top_k_local, show=False)
                plt.tight_layout()
                st.pyplot(fig_wf)
                plt.close(fig_wf)
            except Exception:
                fig_bar, ax_bar = plt.subplots(figsize=(8,5))
                sns.barplot(x='SHAP', y='Feature', data=shap_df.head(top_k_local), orient='h', ax=ax_bar)
                ax_bar.set_title("Top SHAP feature contributions (signed)")
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

        except Exception as e:
            st.error(f"Error preparing local explanation: {e}")

else:
    st.info("Train a model (or upload dataset) to get started.")
    
