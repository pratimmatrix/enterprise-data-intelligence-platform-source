"""
Adaptive Enterprise Data Intelligence Platform
Unified Enterprise Customer Intelligence Dashboard

Author: Pratim Mistry
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
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
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Enterprise Customer Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished enterprise UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .status-high { color: #D32F2F; font-weight: bold; }
    .status-med { color: #F57C00; font-weight: bold; }
    .status-low { color: #388E3C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize Session Singletons
if "predictor" not in st.session_state:
    st.session_state.predictor = ModelPredictor()
if "business_rules" not in st.session_state:
    st.session_state.business_rules = BusinessRuleEngine()
if "insight_engine" not in st.session_state:
    st.session_state.insight_engine = InsightEngine()
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

# ============================================================
# SIDEBAR NAVIGATION & ADMIN CONTROLS
# ============================================================
with st.sidebar:
    st.title("🏦 Platform Portal")
    st.caption("Enterprise AI & Customer Decision Engine")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation Menu",
        ["🎯 Customer Decision Studio", "⚡ Model Management & Adaptive Retraining"]
    )
    
    st.markdown("---")
    if not st.session_state.is_authenticated:
        st.subheader("🔐 Admin Authentication")
        with st.form("admin_login_form"):
            username = st.text_input("Admin ID")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Log In")
            
            if login_btn:
                if AuthManager.verify_credentials(username, password):
                    st.session_state.is_authenticated = True
                    st.success("Admin authorized!")
                    st.rerun()
                else:
                    st.error("Invalid ID / Password.")
    else:
        st.success("🔓 Authenticated as Administrator")
        st.markdown("---")
        st.subheader("⚙️ System Recovery")
        if st.button("🔄 Reset Baseline Model", help="Revert champion model back to pristine baseline"):
            try:
                baseline_path = PROJECT_ROOT / "bank-full.csv"
                if baseline_path.exists():
                    df_base = pd.read_csv(baseline_path, sep=";")
                    reg = SchemaRegistry()
                    reg.register_schema(df_base, version_tag="v1.0", notes="Original Verified Baseline")
                    trainer = AdaptiveModelTrainer()
                    trainer.train_and_evaluate(df_base)
                    st.session_state.predictor = ModelPredictor()
                    st.success("Platform restored to default production baseline!")
                    st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {str(e)}")

        if st.button("Log Out"):
            st.session_state.is_authenticated = False
            st.rerun()

# ============================================================
# VIEW 1: COMPLETE CUSTOMER DECISION STUDIO
# ============================================================
if app_mode == "🎯 Customer Decision Studio":
    st.title("🎯 Customer Conversion Intelligence Studio")
    st.markdown("Real-time predictive scoring, behavioral insights, and automated marketing directives.")
    st.markdown("---")

    with st.expander("📝 Enter Customer Profile & Campaign Outreach Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=38)
            job = st.selectbox("Job Category", [
                "management", "technician", "entrepreneur", "blue-collar", "admin.",
                "services", "retired", "self-employed", "unemployed", "student",
                "housemaid", "unknown"
            ], index=0)
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"], index=0)
            education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"], index=0)

        with col2:
            st.markdown("#### 💳 Financial Profile")
            default = st.selectbox("Has Credit Default?", ["no", "yes"], index=0)
            balance = st.number_input("Average Yearly Balance ($)", value=2500, step=100)
            housing = st.selectbox("Has Housing Loan?", ["yes", "no"], index=0)
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"], index=0)

        with col3:
            st.markdown("#### 📞 Campaign Outreach")
            contact = st.selectbox("Communication Type", ["cellular", "telephone", "unknown"], index=0)
            month = st.selectbox("Last Contact Month", [
                "may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "oct", "sep", "mar", "dec"
            ], index=0)
            day = st.slider("Last Contact Day of Month", 1, 31, 18)
            duration = st.number_input("Last Call Duration (seconds)", min_value=0, value=350, step=10)
            campaign = st.number_input("Contacts in Current Campaign", min_value=1, value=1, step=1)
            pdays = st.number_input("Days Since Prior Campaign (-1 = Never)", min_value=-1, value=-1, step=1)
            previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, value=0, step=1)
            poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "success", "failure", "other"], index=0)

    st.markdown("---")
    
    if st.button("🚀 Run Conversion & Intelligence Prediction", use_container_width=True):
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

            prob_pct = prediction_output["probability_percent"]
            pred_label = prediction_output["prediction"]
            priority = business_output["priority"]
            risk_cat = prediction_output["risk_category"]

            st.markdown("### 📊 Prediction & Decision Dashboard")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric(
                    label="Term Deposit Subscription",
                    value=f"{'✅ YES' if pred_label == 'YES' else '❌ NO'}",
                    delta=f"{prob_pct}% Propensity"
                )
            with kpi2:
                st.metric(
                    label="Conversion Probability",
                    value=f"{prob_pct:.1f}%",
                    delta="Calculated Likelihood"
                )
            with kpi3:
                st.metric(
                    label="Outreach Priority",
                    value=priority,
                    delta=f"Risk: {risk_cat}"
                )
            with kpi4:
                st.metric(
                    label="Expected Value Segment",
                    value=f"{'High Tier' if balance > 5000 else 'Standard Tier'}"
                )

            # Progress Bar for conversion probability
            st.markdown("#### Propensity Distribution")
            st.progress(min(max(float(prob_pct) / 100.0, 0.0), 1.0))

            # Operational Strategy & Directives
            st.markdown("---")
            d_col1, d_col2 = st.columns(2)

            with d_col1:
                st.markdown("### 💼 Operational Directive & Next Best Action")
                if priority == "HIGH":
                    st.success(f"**Action Recommended:** {business_output['recommended_action']}")
                elif priority == "MEDIUM":
                    st.warning(f"**Action Recommended:** {business_output['recommended_action']}")
                else:
                    st.info(f"**Action Recommended:** {business_output['recommended_action']}")

            with d_col2:
                st.markdown("### 💡 Behavioral & Marketing Insights")
                for item in insights:
                    st.markdown(f"• {item}")

        except Exception as e:
            st.error(f"Prediction Pipeline Error: {str(e)}")

# ============================================================
# VIEW 2: AUTHORIZED RETRAINING & SCHEMA STUDIO
# ============================================================
elif app_mode == "⚡ Model Management & Adaptive Retraining":
    st.title("⚡ Dynamic Schema Evolution & Retraining Studio")
    st.markdown("Manage models, upload updated datasets, and monitor dynamic schema adaptations.")
    st.markdown("---")

    if not st.session_state.is_authenticated:
        st.warning("⚠️ Access Restricted. Please log in through the Admin panel on the sidebar.")
    else:
        # Step A: Download Baseline CSV
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
        st.subheader("📤 Upload Candidate Dataset for Defensive Validation & Retraining")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                candidate_df = pd.read_csv(uploaded_file, sep=";")
                if candidate_df.shape[1] == 1:
                    uploaded_file.seek(0)
                    candidate_df = pd.read_csv(uploaded_file, sep=",")

                st.success(f"Candidate dataset parsed successfully! Total Shape: {candidate_df.shape}")

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

                    # Schema Diff Analysis
                    st.markdown("### 🔄 Step 2: Schema Evolution & Diff Analysis")
                    registry = SchemaRegistry()
                    comparator = SchemaComparator(registry)
                    diff = comparator.compare_against_active(candidate_df)

                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.write("**Added Features:**", diff["added_features"] if diff["added_features"] else "None")
                    with d_col2:
                        st.write("**Deleted Features:**", diff["deleted_features"] if diff["deleted_features"] else "None")

                    # Adaptive Retraining Trigger
                    st.markdown("---")
                    if st.button("⚡ Execute Adaptive Retraining & Quality Gate", use_container_width=True):
                        with st.spinner("Retraining candidate model suite dynamically..."):
                            trainer = AdaptiveModelTrainer()
                            training_output = trainer.train_and_evaluate(candidate_df)
                            
                            best_candidate = training_output["best_candidate_name"]
                            best_metrics = training_output["best_candidate_metrics"]

                            st.markdown("### 🏆 Candidate Benchmark Results")
                            candidates_df = pd.DataFrame(training_output["all_candidates"]).T[["roc_auc", "f1", "recall", "precision", "accuracy"]]
                            st.dataframe(candidates_df, use_container_width=True)

                            # Quality Gate Governance
                            st.markdown("### 🛡️ Quality Gate Governance Verdict")
                            quality_gate = QualityGateEngine()
                            gate_verdict = quality_gate.evaluate_candidate(best_candidate, best_metrics)

                            if gate_verdict["decision"] == QualityGateDecision.PROMOTE:
                                st.balloons()
                                st.success(f"🎉 Champion Model Promoted: {best_candidate.upper()}")
                                registry.register_schema(candidate_df, notes=f"Champion: {best_candidate}")
                                st.session_state.predictor = ModelPredictor()
                            else:
                                st.error("❌ Candidate Model Rejected by Quality Gate. Baseline preserved.")
                            
                            st.json(gate_verdict)

            except Exception as ex:
                st.error(f"Execution Error: {str(ex)}")