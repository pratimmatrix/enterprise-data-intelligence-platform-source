# ============================================================
# ENTERPRISE DATA INTELLIGENCE PLATFORM
# STREAMLIT DECISION INTELLIGENCE DASHBOARD
# ============================================================

import sys
from pathlib import Path

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise Data Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    Path.home()
    / "Documents"
    / "models"
    / "random_forest_pipeline.pkl"
)


# ============================================================
# MODEL METRICS
# ============================================================

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
    # BALANCE LOG
    # --------------------------------------------------------

    import math

    customer["balance_log"] = math.log1p(
        abs(customer["balance"])
    )


    # --------------------------------------------------------
    # CAMPAIGN LOG
    # --------------------------------------------------------

    customer["campaign_log"] = math.log1p(
        customer["campaign"]
    )


    # --------------------------------------------------------
    # PREVIOUS CONTACT
    # --------------------------------------------------------

    customer["previous_contact"] = int(
        customer["pdays"] != -1
    )


    # --------------------------------------------------------
    # PREVIOUSLY CONTACTED
    # --------------------------------------------------------

    customer["previously_contacted"] = int(
        customer["previous"] > 0
    )


    # --------------------------------------------------------
    # ZERO BALANCE
    # --------------------------------------------------------

    customer["zero_balance"] = int(
        customer["balance"] == 0
    )


    # --------------------------------------------------------
    # LOAN BURDEN
    # --------------------------------------------------------

    customer["loan_burden"] = int(
        customer["housing"] == "yes"
        and customer["loan"] == "yes"
    )


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
    # UNKNOWN CONTACT
    # --------------------------------------------------------

    customer["contact_unknown"] = int(
        customer["contact"] == "unknown"
    )


    # --------------------------------------------------------
    # PREVIOUS SUCCESS
    # --------------------------------------------------------

    customer["previous_success"] = int(
        customer["poutcome"] == "success"
    )


    return customer


def load_model():

    if not MODEL_PATH.exists():

        return None

    return joblib.load(
        MODEL_PATH
    )


def get_top_features():

    model = load_model()

    if model is None:

        return pd.DataFrame(
            columns=[
                "feature",
                "importance"
            ]
        )

    try:

        preprocessor = (
            model.named_steps["preprocessor"]
        )

        classifier = (
            model.named_steps["classifier"]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            classifier.feature_importances_
        )

        feature_data = sorted(
            zip(
                feature_names,
                importances
            ),
            key=lambda x: x[1],
            reverse=True
        )

        feature_data = feature_data[:10]

        return pd.DataFrame(
            feature_data,
            columns=[
                "feature",
                "importance"
            ]
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "feature",
                "importance"
            ]
        )


def normalize_prediction(prediction):

    if isinstance(
        prediction,
        str
    ):

        value = prediction.upper()

        if value in [
            "YES",
            "NO"
        ]:

            return value

    if prediction in [
        1,
        True
    ]:

        return "YES"

    return "NO"


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Enterprise Data Intelligence Platform"
)

st.caption(
    "AI-powered customer campaign decision intelligence"
)

st.divider()


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.header(
    "👤 Customer Information"
)


with st.form(
    "customer_form"
):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ],
            index=4
        )

        marital = st.selectbox(
            "Marital Status",
            [
                "married",
                "single",
                "divorced"
            ],
            index=0
        )

        education = st.selectbox(
            "Education",
            [
                "primary",
                "secondary",
                "tertiary",
                "unknown"
            ],
            index=2
        )

        default = st.selectbox(
            "Credit Default",
            [
                "no",
                "yes"
            ],
            index=0
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        balance = st.number_input(
            "Account Balance",
            value=1500,
            step=100
        )

        housing = st.selectbox(
            "Housing Loan",
            [
                "yes",
                "no"
            ],
            index=0
        )

        loan = st.selectbox(
            "Personal Loan",
            [
                "no",
                "yes"
            ],
            index=0
        )

        contact = st.selectbox(
            "Contact Channel",
            [
                "cellular",
                "telephone",
                "unknown"
            ],
            index=0
        )

        day = st.number_input(
            "Contact Day",
            min_value=1,
            max_value=31,
            value=15
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        month = st.selectbox(
            "Month",
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec"
            ],
            index=4
        )

        duration = st.number_input(
            "Contact Duration (seconds)",
            min_value=0,
            value=300,
            step=10
        )

        campaign = st.number_input(
            "Campaign Contacts",
            min_value=1,
            value=2,
            step=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=-1,
            value=-1,
            step=1
        )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            value=0,
            step=1
        )


    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        [
            "unknown",
            "failure",
            "other",
            "success"
        ],
        index=0
    )


    st.divider()


    submitted = st.form_submit_button(
        "🚀 Analyze Customer",
        use_container_width=True
    )


# ============================================================
# RUN DECISION ENGINE
# ============================================================

if submitted:

    raw_customer = {

        "age": age,

        "job": job,

        "marital": marital,

        "education": education,

        "default": default,

        "balance": balance,

        "housing": housing,

        "loan": loan,

        "contact": contact,

        "day": day,

        "month": month,

        "duration": duration,

        "campaign": campaign,

        "pdays": pdays,

        "previous": previous,

        "poutcome": poutcome
    }


    customer_data = create_engineered_features(
        raw_customer
    )


    with st.spinner(
        "Running enterprise decision pipeline..."
    ):

        try:

            engine = (
                st.session_state
                .decision_engine
            )

            result = engine.run(
                customer_data
            )

            st.session_state.result = result

            st.session_state.customer = (
                customer_data
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()


# ============================================================
# DISPLAY RESULT
# ============================================================

if "result" in st.session_state:

    result = (
        st.session_state.result
    )

    customer = (
        st.session_state.customer
    )


    st.divider()

    st.header(
        "🎯 Customer Decision"
    )


    # --------------------------------------------------------
    # NORMALIZE VALUES
    # --------------------------------------------------------

    prediction = normalize_prediction(
        result.get(
            "prediction",
            "NO"
        )
    )


    probability = float(
        result.get(
            "probability_percent",
            0
        )
    )


    risk = result.get(
        "risk_category",
        "UNKNOWN"
    )


    priority = result.get(
        "priority",
        "UNKNOWN"
    )


    # --------------------------------------------------------
    # DECISION METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Prediction",
            prediction
        )


    with col2:

        st.metric(
            "Probability",
            f"{probability:.2f}%"
        )


    with col3:

        st.metric(
            "Risk Category",
            risk
        )


    with col4:

        st.metric(
            "Priority",
            priority
        )


    # --------------------------------------------------------
    # BUSINESS DECISION
    # --------------------------------------------------------

    st.subheader(
        "💼 Business Decision"
    )


    st.info(
        result.get(
            "recommended_action",
            "No recommendation available."
        )
    )


    # ========================================================
    # BUSINESS INSIGHTS
    # ========================================================

    st.subheader(
        "💡 Business Insights"
    )


    insights = result.get(
        "insights",
        []
    )


    if insights:

        for insight in insights:

            st.markdown(
                f"• {insight}"
            )

    else:

        st.info(
            "No business insights available."
        )


    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.subheader(
        "👤 Customer Profile"
    )


    profile_col1, profile_col2 = (
        st.columns(2)
    )


    with profile_col1:

        st.markdown(
            f"**Age:** {customer['age']}"
        )

        st.markdown(
            f"**Job:** {customer['job']}"
        )

        st.markdown(
            f"**Education:** {customer['education']}"
        )

        st.markdown(
            f"**Balance:** {customer['balance']}"
        )

        st.markdown(
            f"**Housing Loan:** {customer['housing']}"
        )

        st.markdown(
            f"**Personal Loan:** {customer['loan']}"
        )


    with profile_col2:

        st.markdown(
            f"**Campaign Contacts:** "
            f"{customer['campaign']}"
        )

        st.markdown(
            f"**Previous Contacts:** "
            f"{customer['previous']}"
        )

        st.markdown(
            f"**Previous Outcome:** "
            f"{customer['poutcome']}"
        )

        st.markdown(
            f"**Contact Channel:** "
            f"{customer['contact']}"
        )

        st.markdown(
            f"**Duration:** "
            f"{customer['duration']} seconds"
        )

        st.markdown(
            f"**Campaign Intensity:** "
            f"{customer['campaign_intensity']}"
        )

        st.markdown(
            f"**Age Group:** "
            f"{customer['age_group']}"
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🤖 Model Information"
    )


    model_col1, model_col2 = (
        st.columns(2)
    )


    with model_col1:

        st.markdown(
            f"**Model:** {MODEL_NAME}"
        )

        st.markdown(
            f"**Pipeline:** {PIPELINE_NAME}"
        )

        st.markdown(
            f"**Selection:** {SELECTION_METHOD}"
        )


    with model_col2:

        st.markdown(
            f"**ROC-AUC:** {ROC_AUC:.4f}"
        )

        st.markdown(
            f"**F1 Score:** {F1_SCORE:.4f}"
        )


    # ========================================================
    # TOP MODEL FEATURES
    # ========================================================

    st.subheader(
        "📈 Top Model Features"
    )


    feature_df = get_top_features()


    if not feature_df.empty:

        chart_df = (
            feature_df
            .set_index("feature")
        )

        st.bar_chart(
            chart_df[
                "importance"
            ]
        )


        with st.expander(
            "View feature importance values"
        ):

            display_df = feature_df.copy()

            display_df[
                "importance"
            ] = display_df[
                "importance"
            ].round(4)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Feature importance is unavailable."
        )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.subheader(
        "🔍 Model Explanation"
    )


    explanations = result.get(
        "explanations",
        []
    )


    if explanations:

        for explanation in explanations:

            st.markdown(
                f"• {explanation}"
            )

    else:

        st.markdown(
            f"The model predicts "
            f"**{prediction}** with a probability "
            f"of **{probability:.2f}%**."
        )

        st.markdown(
            f"The resulting model risk category "
            f"is **{risk}**."
        )

        st.markdown(
            f"• The customer has "
            f"{customer['previous']} previous "
            f"campaign contact(s)."
        )

        st.markdown(
            f"• The previous campaign outcome "
            f"is **{customer['poutcome']}**."
        )

        st.markdown(
            f"• The customer received "
            f"{customer['campaign']} current "
            f"campaign contact(s)."
        )

        st.markdown(
            f"• The customer was contacted through "
            f"the **{customer['contact']}** channel."
        )

        st.markdown(
            f"• Current contact duration is "
            f"{customer['duration']} seconds."
        )

        st.markdown(
            f"• Customer age is "
            f"{customer['age']}."
        )

        st.markdown(
            f"• Customer account balance is "
            f"{customer['balance']}."
        )


    # ========================================================
    # TOP FEATURES IN EXPLANATION
    # ========================================================

    top_features = result.get(
        "top_features",
        []
    )


    if top_features:

        st.subheader(
            "🔬 Global Feature Importance"
        )


        for number, feature in enumerate(
            top_features,
            start=1
        ):

            if isinstance(
                feature,
                dict
            ):

                feature_name = feature.get(
                    "feature",
                    "Unknown"
                )

                importance = float(
                    feature.get(
                        "importance",
                        0
                    )
                )

            else:

                feature_name = str(
                    feature
                )

                importance = 0


            st.markdown(
                f"**{number}. "
                f"{feature_name}** — "
                f"{importance:.4f}"
            )


    # ========================================================
    # FINAL DECISION SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Final Decision Summary"
    )


    summary = pd.DataFrame(
        {
            "Decision Component": [
                "Prediction",
                "Probability",
                "Risk Category",
                "Priority",
                "Recommended Action"
            ],
            "Result": [
                prediction,
                f"{probability:.2f}%",
                risk,
                priority,
                result.get(
                    "recommended_action",
                    "N/A"
                )
            ]
        }
    )


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise Data Intelligence Platform | "
    "ML + Business Rules + Explainability"
)