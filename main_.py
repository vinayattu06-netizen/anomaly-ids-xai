import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================================================
# STREAMLIT PAGE CONFIG
# ==================================================================================
st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")
st.title(" Intrusion Detection System + XAI Dashboard")
st.write(
    "This tool loads the KDD dataset, trains a Random Forest model, "
    "detects attacks, and explains them using SHAP."
)

# ==================================================================================
# LOAD & PREPROCESS DATA
# ==================================================================================
@st.cache_data
def load_data():
    column_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
        'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
        'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
        'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','target'
    ]

    df = pd.read_csv("KDDTrain+.txt", header=None)
    df.columns = column_names + ["target_num"]

    # Binary label: normal = 0, attack = 1
    df["label"] = df["target"].apply(lambda x: 0 if x == "normal" else 1)

    # Encode categorical columns
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=["target", "target_num", "label"])
    y = df_encoded["label"]

    return df, df_encoded, X, y


with st.spinner("Loading dataset..."):
    df, df_encoded, X, y = load_data()
st.success("Dataset loaded successfully!")

# ==================================================================================
# TRAIN MODEL
# ==================================================================================
@st.cache_resource
def train_pipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    df_test_reset = df.loc[X_test.index].reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)

    return model, X_train, X_test_reset, y_test, y_pred, y_proba, acc, report, matrix, df_test_reset


# ==================================================================================
# SESSION STATE TO KEEP DASHBOARD FROM RESETTING
# ==================================================================================
if "run_model" not in st.session_state:
    st.session_state["run_model"] = False

if st.button("🚀 Train Model & Run Detection", type="primary"):
    st.session_state["run_model"] = True

if st.session_state["run_model"]:
    with st.spinner("Training model..."):
        (
            model,
            X_train,
            X_test_reset,
            y_test,
            y_pred,
            y_proba,
            acc,
            report,
            matrix,
            df_test_reset,
        ) = train_pipeline(X, y)

    st.success("Model training completed!")

    # ==================================================================================
    # MODEL METRICS SECTION
    # ==================================================================================
    st.subheader("📊 Model Performance Metrics")

    col1, col2 = st.columns(2)

    # Accuracy
    with col1:
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    with col2:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            cmap="Blues",
            fmt="d",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=ax,
        )
        st.pyplot(fig)

    # ==================================================================================
    # FLAGGED ATTACKS
    # ==================================================================================
    st.subheader("🚨 Detected Attacks (Predicted as Attack = 1)")
    flagged_indices = [i for i, pred in enumerate(y_pred) if pred == 1]

    st.write(f"**Total flagged attacks:** `{len(flagged_indices)}`")

    if len(flagged_indices) == 0:
        st.info("No attacks detected. Try another dataset split.")
    else:
        import shap

        # ==================================================================================
        # SHAP GLOBAL SUMMARY
        # ==================================================================================
        st.subheader("🧠 SHAP Explainability (Global Feature Impact)")

        @st.cache_resource
        def get_shap_explainer(_model, _X_train):
            # leading underscores -> Streamlit won't hash these args
            explainer = shap.TreeExplainer(_model)
            background = _X_train.sample(
                n=min(500, len(_X_train)), random_state=42
            )
            return explainer, background

        explainer, background = get_shap_explainer(model, X_train)

        # Take up to first 100 flagged samples
        sample_data = X_test_reset.iloc[flagged_indices[:100]]
        shap_values = explainer(sample_data)

        # For binary classifier, SHAP returns shape (n_samples, n_features, 2)
        # We focus on class 1 (attack) for global summary
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_global = shap_values.values[:, :, 1]
        else:
            shap_global = shap_values.values

        shap.summary_plot(shap_global, sample_data, show=False)
        fig_shap = plt.gcf()
        st.pyplot(fig_shap)

        # ==================================================================================
        # PER-ATTACK SHAP EXPLANATION
        # ==================================================================================
        st.subheader("🔍 Detailed SHAP Explanation for a Specific Attack")

        attack_choice = st.selectbox("Select attack index:", flagged_indices)

        sample = X_test_reset.iloc[[attack_choice]]
        true_label = df_test_reset.loc[attack_choice, "target"]

        # Compute SHAP for the selected sample
        instance_shap = explainer(sample)

        # For binary classifier, shape is (1, n_features, 2)
        # Get predicted class and take that slice
        pred_class = model.predict(sample)[0]  # 0 or 1
        if instance_shap.values.ndim == 3:
            shap_vals = instance_shap.values[0][:, pred_class]
        else:
            shap_vals = instance_shap.values[0]

        proba = model.predict_proba(sample)[0][1]  # P(attack)

        level = "HIGH" if proba > 0.85 else "MEDIUM" if proba > 0.5 else "LOW"

        st.metric("Prediction", "Attack" if pred_class == 1 else "Normal")
        st.metric("Attack Probability", f"{proba:.4f}")
        st.metric("Risk Level", level)
        st.write(f"**Ground truth:** `{true_label}`")

        shap_df = pd.DataFrame({
            "Feature": sample.columns,
            "Value": sample.values[0],
            "SHAP": shap_vals,
            "Impact": ["Increases risk" if x > 0 else "Decreases risk" for x in shap_vals],
            "Abs": np.abs(shap_vals),
        }).sort_values("Abs", ascending=False)

        st.write("### Top Contributing Features")
        st.dataframe(shap_df[["Feature", "Value", "SHAP", "Impact"]].head(10))

        st.write("### Waterfall Plot")
        # For waterfall, also pass the slice for the predicted class
        if instance_shap.values.ndim == 3:
            shap.plots.waterfall(instance_shap[0, :, pred_class], show=False)
        else:
            shap.plots.waterfall(instance_shap[0], show=False)
        fig_wf = plt.gcf()
        st.pyplot(fig_wf)

else:
    st.info("Click **Train & Run Detection** to start.")
