"""
Adaptive Enterprise Data Intelligence Platform
Streamlit Dual-Access Decision & Retraining Dashboard

Author: Pratim Mistry
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT ROOT PATH RESOLUTION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth.auth_manager import AuthManager
from src.modeling.ModelPredictor import ModelPredictor
from src.business_rules.BusinessRuleEngine import BusinessRuleEngine
from src.insights.InsightEngine import InsightEngine
from src.gatekeeper.dataset_gatekeeper import DatasetGatekeeper, GatekeeperDecision
from src.schema.schema_registry import SchemaRegistry
from src.schema.schema_comparator import SchemaComparator
from src.training.adaptive_trainer import AdaptiveModelTrainer
from src.training.quality_gate import QualityGateEngine, QualityGateDecision

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Adaptive Enterprise Data Platform",
    page_icon="🧠",
    layout="wide"
)

# Initialize singletons
if "predictor" not in st.session_state:
    st.session_state.predictor = ModelPredictor()
if "business_rules" not in st.session_state:
    st.session_state.business_rules = BusinessRuleEngine()
if "insight_engine" not in st.session_state:
    st.session_state.insight_engine = InsightEngine()
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# ============================================================
# SIDEBAR: NAVIGATION & AUTHENTICATION
# ============================================================
with st.sidebar:
    st.title("🛡️ Platform Control")
    st.markdown("---")
    
    app_mode = st.radio(
        "Select Operation Mode",
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
        
        # System Reset Action
        st.markdown("---")
        st.subheader("⚙️ System Governance")
        if st.button("🔄 Reset to Original Baseline", help="Instantly restore baseline models and active schema"):
            try:
                # Re-register original baseline
                baseline_path = PROJECT_ROOT / "bank-full.csv"
                if baseline_path.exists():
                    df_base = pd.read_csv(baseline_path, sep=";")
                    reg = SchemaRegistry()
                    reg.register_schema(df_base, version_tag="v1.0", notes="Original Verified Baseline")
                    trainer = AdaptiveModelTrainer()
                    trainer.train_and_evaluate(df_base)
                    st.session_state.predictor = ModelPredictor()
                    st.success("Platform restored to pristine baseline state!")
                    st.rerun()
            except Exception as e:
                st.error(f"Reset Error: {str(e)}")

        if st.button("Log Out"):
            st.session_state.is_authenticated = False
            st.rerun()

# ============================================================
# VIEW 1: PUBLIC DECISION STUDIO
# ============================================================
if app_mode == "🎯 Public Decision Studio":
    st.title("🎯 Enterprise Customer Decision Studio")
    st.caption("Real-Time Machine Learning Inference & Heuristic Business Optimization")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job Type", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"
        ], index=4)
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"], index=0)
        education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"], index=2)

    with col2:
        st.subheader("💳 Financial Health")
        default = st.selectbox("Credit Default History", ["no", "yes"], index=0)
        balance = st.number_input("Account Balance ($)", value=1500, step=100)
        housing = st.selectbox("Housing Loan", ["yes", "no"], index=0)
        loan = st.selectbox("Personal Loan", ["no", "yes"], index=0)

    with col3:
        st.subheader("📞 Campaign Outreach")
        contact = st.selectbox("Contact Communication Channel", ["cellular", "telephone", "unknown"], index=0)
        month = st.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ], index=4)
        day = st.slider("Last Contact Day", 1, 31, 15)
        duration = st.number_input("Call Duration (seconds)", min_value=0, value=300, step=10)
        campaign = st.number_input("Current Campaign Contacts", min_value=1, value=2, step=1)
        pdays = st.number_input("Days Since Prior Campaign (-1 = Never)", min_value=-1, value=-1, step=1)
        previous = st.number_input("Previous Contact Count", min_value=0, value=0, step=1)
        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"], index=0)

    st.markdown("---")
    if st.button("🚀 Evaluate Customer Conversion Probability", use_container_width=True):
        customer_payload = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
        }

        try:
            prediction_output = st.session_state.predictor.predict(customer_payload)
            business_output = st.session_state.business_rules.evaluate(prediction_output)
            insights = st.session_state.insight_engine.generate_insights(customer_payload, business_output)

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Conversion Prediction", prediction_output["prediction"])
            with res_col2:
                st.metric("Conversion Probability", f"{prediction_output['probability_percent']}%")
            with res_col3:
                st.metric("Risk Priority", business_output["priority"])
            with res_col4:
                st.metric("Target Confidence", prediction_output["risk_category"])

            st.markdown("### 💼 Operational Directive")
            st.info(business_output["recommended_action"])

            st.markdown("### 💡 Marketing & Strategic Insights")
            for ins in insights:
                st.markdown(f"• {ins}")

        except Exception as e:
            st.error(f"Inference Engine Error: {str(e)}")

# ============================================================
# VIEW 2: AUTHORIZED RETRAINING & SCHEMA STUDIO
# ============================================================
elif app_mode == "⚡ Authorized Retraining & Schema Studio":
    st.title("⚡ Dynamic Schema Evolution & Retraining Studio")
    st.caption("Defensive Dataset Ingestion, Dynamic Schema Adaptation, and Automated Quality Gates")
    st.markdown("---")

    if not st.session_state.is_authenticated:
        st.warning("⚠️ Access Restricted. Please authenticate via the admin panel on the sidebar.")
    else:
        # 1. Download Baseline Dataset
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

                # Defensive Gatekeeper Evaluation
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

                    # Step 2: Schema Evolution
                    st.markdown("### 🔄 Step 2: Schema Evolution & Diff Analysis")
                    registry = SchemaRegistry()
                    comparator = SchemaComparator(registry)
                    diff = comparator.compare_against_active(candidate_df)

                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.write("**Added Features:**", diff["added_features"] if diff["added_features"] else "None")
                    with d_col2:
                        st.write("**Deleted Features:**", diff["deleted_features"] if diff["deleted_features"] else "None")

                    # Step 3: Trigger Automated Retraining
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

                            # Step 4: Quality Gate Check
                            st.markdown("### 🛡️ Quality Gate Governance Verdict")
                            quality_gate = QualityGateEngine()
                            gate_verdict = quality_gate.evaluate_candidate(best_candidate, best_metrics)

                            if gate_verdict["decision"] == QualityGateDecision.PROMOTE:
                                st.balloons()
                                st.success(f"🎉 Champion Model Promoted: {best_candidate.upper()}")
                                registry.register_schema(candidate_df, notes=f"Champion: {best_candidate}")
                                st.session_state.predictor = ModelPredictor() # Refresh inference engine
                            else:
                                st.error("❌ Candidate Model Rejected by Quality Gate. Baseline preserved.")
                            
                            st.json(gate_verdict)

            except Exception as ex:
                st.error(f"Execution Error: {str(ex)}")