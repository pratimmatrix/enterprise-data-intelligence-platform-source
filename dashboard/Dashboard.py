# ============================================================
# ENTERPRISE DATA INTELLIGENCE PLATFORM
# STREAMLIT DECISION INTELLIGENCE DASHBOARD (INTEGRATED)
# ============================================================

import sys
from pathlib import Path
import math

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.business_rules.DecisionEngine import DecisionEngine
from src.auth.auth_manager import AuthManager
from src.gatekeeper.dataset_gatekeeper import DatasetGatekeeper, GatekeeperDecision
from src.schema.schema_registry import SchemaRegistry
from src.schema.schema_comparator import SchemaComparator
from src.training.adaptive_trainer import AdaptiveModelTrainer
from src.training.quality_gate import QualityGateEngine, QualityGateDecision

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# MODEL PATH & METRICS
# ============================================================

MODEL_PATH = (
    Path.home()
    / "Documents"
    / "models"
    / "random_forest_pipeline.pkl"
)

MODEL_NAME = "Random Forest Classifier"
PIPELINE_NAME = "Preprocessor + Random Forest"
SELECTION_METHOD = "F1 Score"

ROC_AUC = 0.7946
F1_SCORE = 0.4297

# ============================================================
# SESSION STATE
# ============================================================

if "decision_engine" not in st.session_state:
    st.session_state.decision_engine = DecisionEngine()

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_engineered_features(customer):
    """
    Create the same engineered features used by the
    Enterprise Data Intelligence Platform.
    """
    customer = customer.copy()

    # --------------------------------------------------------
    # AGE GROUP
    # --------------------------------------------------------
    age = customer["age"]
    if age <= 20:
        customer["age_group"] = "0-20"
    elif age <= 30:
        customer["age_group"] = "21-30"
    elif age <= 40:
        customer["age_group"] = "31-40"
    elif age <= 50:
        customer["age_group"] = "41-50"
    elif age <= 60:
        customer["age_group"] = "51-60"
    else:
        customer["age_group"] = "61+"

    # --------------------------------------------------------
    # BALANCE LOG & CAMPAIGN LOG
    # --------------------------------------------------------
    customer["balance_log"] = math.log1p(abs(customer["balance"]))
    customer["campaign_log"] = math.log1p(customer["campaign"])

    # --------------------------------------------------------
    # CONTACT & BALANCE FLAGS
    # --------------------------------------------------------
    customer["previous_contact"] = int(customer["pdays"] != -1)
    customer["previously_contacted"] = int(customer["previous"] > 0)
    customer["zero_balance"] = int(customer["balance"] == 0)
    customer["loan_burden"] = int(customer["housing"] == "yes" and customer["loan"] == "yes")

    # --------------------------------------------------------
    # CAMPAIGN INTENSITY
    # --------------------------------------------------------
    campaign = customer["campaign"]
    if campaign <= 2:
        customer["campaign_intensity"] = "low"
    elif campaign <= 5:
        customer["campaign_intensity"] = "medium"
    else:
        customer["campaign_intensity"] = "high"

    # --------------------------------------------------------
    # UNKNOWN & PREVIOUS SUCCESS
    # --------------------------------------------------------
    customer["contact_unknown"] = int(customer["contact"] == "unknown")
    customer["previous_success"] = int(customer["poutcome"] == "success")

    return customer


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_top_features():
    model = load_model()
    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        feature_data = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        feature_data = feature_data[:10]

        return pd.DataFrame(feature_data, columns=["feature", "importance"])
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def normalize_prediction(prediction):
    if isinstance(prediction, str):
        value = prediction.upper()
        if value in ["YES", "NO"]:
            return value
    if prediction in [1, True]:
        return "YES"
    return "NO"


# ============================================================
# SIDEBAR CONTROLS & AUTHENTICATION
# ============================================================

with st.sidebar:
    st.title("🛡️ Platform Control")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        ["🎯 Public Decision Studio", "⚡ Authorized Retraining & Schema Studio"]
    )
    
    st.markdown("---")
    if not st.session_state.is_authenticated:
        st.subheader("🔐 Admin Access")
        with st.form("admin_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Authenticate")
            
            if login_btn:
                if AuthManager.verify_credentials(username, password):
                    st.session_state.is_authenticated = True
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    else:
        st.success("🔓 Authenticated as Administrator")
        
        st.markdown("---")
        st.subheader("⚙️ System Governance")
        if st.button("🔄 Reset Baseline Model", help="Revert model and schema back to default verified state"):
            try:
                baseline_path = PROJECT_ROOT / "bank-full.csv"
                if baseline_path.exists():
                    df_base = pd.read_csv(baseline_path, sep=";")
                    reg = SchemaRegistry()
                    reg.register_schema(df_base, version_tag="v1.0", notes="Original Verified Baseline")
                    trainer = AdaptiveModelTrainer()
                    trainer.train_and_evaluate(df_base)
                    st.session_state.decision_engine = DecisionEngine()
                    st.success("Platform restored to pristine baseline state!")
                    st.rerun()
            except Exception as e:
                st.error(f"Reset Error: {str(e)}")

        if st.button("Log Out"):
            st.session_state.is_authenticated = False
            st.rerun()

# ============================================================
# VIEW 1: ORIGINAL CUSTOMER DECISION STUDIO
# ============================================================

if app_mode == "🎯 Public Decision Studio":

    st.title("📊 Enterprise Data Intelligence Platform")
    st.caption("AI-powered customer campaign decision intelligence")
    st.divider()

    # --------------------------------------------------------
    # CUSTOMER INPUT FORM
    # --------------------------------------------------------
    st.header("👤 Customer Information")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        # COLUMN 1
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            job = st.selectbox(
                "Job",
                [
                    "admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown"
                ],
                index=4
            )
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"], index=0)
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"], index=2)
            default = st.selectbox("Credit Default", ["no", "yes"], index=0)

        # COLUMN 2
        with col2:
            balance = st.number_input("Account Balance", value=1500, step=100)
            housing = st.selectbox("Housing Loan", ["yes", "no"], index=0)
            loan = st.selectbox("Personal Loan", ["no", "yes"], index=0)
            contact = st.selectbox("Contact Channel", ["cellular", "telephone", "unknown"], index=0)
            day = st.number_input("Contact Day", min_value=1, max_value=31, value=15)

        # COLUMN 3
        with col3:
            month = st.selectbox(
                "Month",
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                index=4
            )
            duration = st.number_input("Contact Duration (seconds)", min_value=0, value=300, step=10)
            campaign = st.number_input("Campaign Contacts", min_value=1, value=2, step=1)
            pdays = st.number_input("Days Since Previous Contact", min_value=-1, value=-1, step=1)
            previous = st.number_input("Previous Contacts", min_value=0, value=0, step=1)

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "failure", "other", "success"],
            index=0
        )

        st.divider()

        submitted = st.form_submit_button(
            "🚀 Analyze Customer",
            use_container_width=True
        )

    # --------------------------------------------------------
    # EXECUTION PIPELINE
    # --------------------------------------------------------
    if submitted:
        raw_customer = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
        }

        customer_data = create_engineered_features(raw_customer)

        with st.spinner("Running enterprise decision pipeline..."):
            try:
                engine = st.session_state.decision_engine
                result = engine.run(customer_data)
                st.session_state.result = result
                st.session_state.customer = customer_data
            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.stop()

    # --------------------------------------------------------
    # ORIGINAL RESULTS & VISUALIZATIONS
    # --------------------------------------------------------
    if "result" in st.session_state:
        result = st.session_state.result
        customer = st.session_state.customer

        st.divider()
        st.header("🎯 Customer Decision")

        prediction = normalize_prediction(result.get("prediction", "NO"))
        probability = float(result.get("probability_percent", 0))
        risk = result.get("risk_category", "UNKNOWN")
        priority = result.get("priority", "UNKNOWN")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Probability", f"{probability:.2f}%")
        with col3:
            st.metric("Risk Category", risk)
        with col4:
            st.metric("Priority", priority)

        st.subheader("💼 Business Decision")
        st.info(result.get("recommended_action", "No recommendation available."))

        st.subheader("💡 Business Insights")
        insights = result.get("insights", [])
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.info("No business insights available.")

        st.subheader("👤 Customer Profile")
        profile_col1, profile_col2 = st.columns(2)
        with profile_col1:
            st.markdown(f"**Age:** {customer['age']}")
            st.markdown(f"**Job:** {customer['job']}")
            st.markdown(f"**Education:** {customer['education']}")
            st.markdown(f"**Balance:** {customer['balance']}")
            st.markdown(f"**Housing Loan:** {customer['housing']}")
            st.markdown(f"**Personal Loan:** {customer['loan']}")
        with profile_col2:
            st.markdown(f"**Campaign Contacts:** {customer['campaign']}")
            st.markdown(f"**Previous Contacts:** {customer['previous']}")
            st.markdown(f"**Previous Outcome:** {customer['poutcome']}")
            st.markdown(f"**Contact Channel:** {customer['contact']}")
            st.markdown(f"**Duration:** {customer['duration']} seconds")
            st.markdown(f"**Campaign Intensity:** {customer['campaign_intensity']}")
            st.markdown(f"**Age Group:** {customer['age_group']}")

        st.subheader("🤖 Model Information")
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.markdown(f"**Model:** {MODEL_NAME}")
            st.markdown(f"**Pipeline:** {PIPELINE_NAME}")
            st.markdown(f"**Selection:** {SELECTION_METHOD}")
        with model_col2:
            st.markdown(f"**ROC-AUC:** {ROC_AUC:.4f}")
            st.markdown(f"**F1 Score:** {F1_SCORE:.4f}")

        st.subheader("📈 Top Model Features")
        feature_df = get_top_features()
        if not feature_df.empty:
            chart_df = feature_df.set_index("feature")
            st.bar_chart(chart_df["importance"])

            with st.expander("View feature importance values"):
                display_df = feature_df.copy()
                display_df["importance"] = display_df["importance"].round(4)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance is unavailable.")

        st.subheader("🔍 Model Explanation")
        explanations = result.get("explanations", [])
        if explanations:
            for explanation in explanations:
                st.markdown(f"• {explanation}")
        else:
            st.markdown(f"The model predicts **{prediction}** with a probability of **{probability:.2f}%**.")
            st.markdown(f"The resulting model risk category is **{risk}**.")
            st.markdown(f"• The customer has {customer['previous']} previous campaign contact(s).")
            st.markdown(f"• The previous campaign outcome is **{customer['poutcome']}**.")
            st.markdown(f"• The customer received {customer['campaign']} current campaign contact(s).")
            st.markdown(f"• The customer was contacted through the **{customer['contact']}** channel.")
            st.markdown(f"• Current contact duration is {customer['duration']} seconds.")
            st.markdown(f"• Customer age is {customer['age']}.")
            st.markdown(f"• Customer account balance is {customer['balance']}.")

        top_features = result.get("top_features", [])
        if top_features:
            st.subheader("🔬 Global Feature Importance")
            for number, feature in enumerate(top_features, start=1):
                if isinstance(feature, dict):
                    feature_name = feature.get("feature", "Unknown")
                    importance = float(feature.get("importance", 0))
                else:
                    feature_name = str(feature)
                    importance = 0
                st.markdown(f"**{number}. {feature_name}** — {importance:.4f}")

        st.divider()
        st.subheader("📋 Final Decision Summary")
        summary = pd.DataFrame({
            "Decision Component": ["Prediction", "Probability", "Risk Category", "Priority", "Recommended Action"],
            "Result": [prediction, f"{probability:.2f}%", risk, priority, result.get("recommended_action", "N/A")]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Enterprise Data Intelligence Platform | ML + Business Rules + Explainability")

# ============================================================
# VIEW 2: AUTHORIZED RETRAINING & SCHEMA STUDIO
# ============================================================

elif app_mode == "⚡ Authorized Retraining & Schema Studio":
    st.title("⚡ Dynamic Schema Evolution & Retraining Studio")
    st.caption("Defensive Ingestion, Schema Evolution Diffing, and Quality Gate Governance")
    st.markdown("---")

    if not st.session_state.is_authenticated:
        st.warning("⚠️ Access Restricted. Please authenticate via the admin panel on the sidebar.")
    else:
        data_path = PROJECT_ROOT / "bank-full.csv"
        if data_path.exists():
            with open(data_path, "rb") as f:
                st.download_button(
                    label="📥 Download Current Baseline Dataset (bank-full.csv)",
                    data=f,
                    file_name="bank-full.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        st.subheader("📤 Upload Candidate Dataset for Ingestion & Retraining")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                candidate_df = pd.read_csv(uploaded_file, sep=";")
                if candidate_df.shape[1] == 1:
                    uploaded_file.seek(0)
                    candidate_df = pd.read_csv(uploaded_file, sep=",")

                st.success(f"Candidate dataset parsed successfully! Shape: {candidate_df.shape}")

                # 1. Defensive Gatekeeper
                gatekeeper = DatasetGatekeeper()
                gate_report = gatekeeper.evaluate_candidate(candidate_df)

                st.markdown("### 🛡️ Step 1: Gatekeeper Verification Report")
                if gate_report["decision"] == GatekeeperDecision.REJECT:
                    st.error("❌ Dataset REJECTED by Defensive Gatekeeper!")
                    for reason in gate_report["reasons"]:
                        st.write(f"• {reason}")
                else:
                    st.success("✅ Dataset ACCEPTED for Retraining Pipeline.")
                    st.json(gate_report)

                    # 2. Schema Evolution
                    st.markdown("### 🔄 Step 2: Schema Evolution & Diff Analysis")
                    registry = SchemaRegistry()
                    comparator = SchemaComparator(registry)
                    diff = comparator.compare_against_active(candidate_df)

                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.write("**Added Features:**", diff["added_features"] if diff["added_features"] else "None")
                    with d_col2:
                        st.write("**Deleted Features:**", diff["deleted_features"] if diff["deleted_features"] else "None")

                    # 3. Dynamic Retraining
                    st.markdown("---")
                    if st.button("⚡ Execute Adaptive Retraining & Champion Evaluation", use_container_width=True):
                        with st.spinner("Retraining candidate model suite dynamically..."):
                            trainer = AdaptiveModelTrainer()
                            training_output = trainer.train_and_evaluate(candidate_df)
                            
                            best_candidate = training_output["best_candidate_name"]
                            best_metrics = training_output["best_candidate_metrics"]

                            st.markdown("### 🏆 Candidate Benchmark Results")
                            candidates_df = pd.DataFrame(training_output["all_candidates"]).T[["roc_auc", "f1", "recall", "precision", "accuracy"]]
                            st.dataframe(candidates_df, use_container_width=True)

                            # 4. Quality Gate Check
                            st.markdown("### 🛡️ Quality Gate Governance Verdict")
                            quality_gate = QualityGateEngine()
                            gate_verdict = quality_gate.evaluate_candidate(best_candidate, best_metrics)

                            if gate_verdict["decision"] == QualityGateDecision.PROMOTE:
                                st.balloons()
                                st.success(f"🎉 Champion Model Promoted: {best_candidate.upper()}")
                                registry.register_schema(candidate_df, notes=f"Champion: {best_candidate}")
                                st.session_state.decision_engine = DecisionEngine()
                            else:
                                st.error("❌ Candidate Model Rejected by Quality Gate. Baseline preserved.")
                            
                            st.json(gate_verdict)

            except Exception as ex:
                st.error(f"Execution Error: {str(ex)}")