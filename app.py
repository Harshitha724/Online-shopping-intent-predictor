"""
Online Shopper Purchase Intention Dashboard
--------------------------------------------
A Streamlit web app: user fills in details about a website visitor session,
clicks Predict, and sees:
  1. Whether the model thinks they'll purchase (yes/no + probability)
  2. A SHAP waterfall plot explaining WHY the model made that prediction

To run: open a terminal in this folder and type:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import urllib.request

# =========================================================
# 1. PAGE SETUP -- this just controls the browser tab title/layout
# =========================================================
st.set_page_config(page_title="Purchase Intention Predictor", layout="centered")
st.title("🛒 Online Shopper Purchase Intention Predictor")
st.write(
    "Fill in details about a website visit below, then click **Predict** to see "
    "whether the model thinks this session will end in a purchase — and why."
)

# =========================================================
# 2. DOWNLOAD LARGE FILES FROM GITHUB RELEASES (if not already present)
# These files are too large for the normal repo, so they're hosted as
# GitHub Release attachments instead. This downloads them once, the first
# time the app runs, and reuses the local copy after that.
# =========================================================
MODEL_URL = "https://github.com/Harshitha724/Online-shopping-intent-predictor/releases/download/v1.0/final_model.pkl"
EXPLAINER_URL = "https://github.com/Harshitha724/Online-shopping-intent-predictor/releases/download/v1.0/shap_explainer.pkl"

def download_if_missing(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} (first run only)..."):
            urllib.request.urlretrieve(url, filename)

download_if_missing(MODEL_URL, "final_model.pkl")
download_if_missing(EXPLAINER_URL, "shap_explainer.pkl")

# =========================================================
# 3. LOAD THE SAVED MODEL + SHAP EXPLAINER (done once when app starts)
# =========================================================
@st.cache_resource  # tells Streamlit: only load these once, not on every click
def load_model_and_explainer():
    model = joblib.load("final_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, explainer

model, explainer = load_model_and_explainer()

# Grab the exact column structure the model was trained on, so we build
# inputs in the exact same shape/order the model expects.
template = pd.read_csv("X_train.csv", nrows=1)
feature_columns = template.columns.tolist()

# =========================================================
# 3. INPUT FORM -- the boxes/sliders the user interacts with
# =========================================================
st.header("Session Details")

col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input("Administrative pages visited", 0, 30, 2)
    administrative_duration = st.number_input("Administrative page time (seconds)", 0, 3000, 60)
    informational = st.number_input("Informational pages visited", 0, 30, 1)
    informational_duration = st.number_input("Informational page time (seconds)", 0, 3000, 30)
    product_related = st.number_input("Product pages visited", 0, 300, 20)
    product_related_duration = st.number_input("Product page time (seconds)", 0, 20000, 600)

with col2:
    exit_rate = st.slider("Exit Rate", 0.0, 0.2, 0.02, step=0.001)
    page_values = st.slider("Page Values", 0.0, 300.0, 5.0, step=0.5)
    special_day = st.slider("Special Day closeness (0=far, 1=very close)", 0.0, 1.0, 0.0, step=0.1)
    weekend = st.checkbox("Session on a weekend?")
    month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul",
                                     "Aug", "Sep", "Oct", "Nov", "Dec"])
    visitor_type = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])

st.caption(
    "Note: Browser, Operating System, and Traffic Type are kept at their most common "
    "values for this demo, to keep the form simple."
)

# =========================================================
# 4. BUILD THE INPUT ROW -- convert form answers into the exact
#    same format (60 columns, one-hot encoded) the model was trained on
# =========================================================
def build_input_row():
    # Start with a single row of all zeros, using the same columns as training data
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # Fill in the numeric features directly
    row["Administrative"] = administrative
    row["Administrative_Duration"] = administrative_duration
    row["Informational"] = informational
    row["Informational_Duration"] = informational_duration
    row["ProductRelated"] = product_related
    row["ProductRelated_Duration"] = product_related_duration
    row["ExitRates"] = exit_rate
    row["PageValues"] = page_values
    row["SpecialDay"] = special_day
    row["Weekend"] = int(weekend)

    # Engineered feature: matches how we built it in preprocessing
    row["IsHolidaySeason"] = 1 if month in ["Nov", "Dec"] else 0

    # One-hot categorical columns: set the matching dummy column to 1 if it exists
    # (Feb and New_Visitor were the "baseline" categories dropped during encoding,
    # so if the user picks those, we correctly leave everything at 0)
    month_col = f"Month_{month}"
    if month_col in row.columns:
        row[month_col] = 1

    visitor_col = f"VisitorType_{visitor_type}"
    if visitor_col in row.columns:
        row[visitor_col] = 1

    return row

# =========================================================
# 5. PREDICT BUTTON -- everything below only runs when clicked
# =========================================================
if st.button("Predict", type="primary"):
    input_row = build_input_row()

    # Get prediction + probability
    prediction = model.predict(input_row)[0]
    probability = model.predict_proba(input_row)[0][1]  # probability of class 1 (purchase)

    st.header("Result")
    if prediction == 1:
        st.success(f"✅ Likely to PURCHASE — probability: {probability:.1%}")
    else:
        st.error(f"❌ Unlikely to purchase — probability: {probability:.1%}")

    # =========================================================
    # 6. SHAP EXPLANATION -- why did the model decide this?
    # =========================================================
    st.header("Why did the model predict this?")

    shap_values = explainer.shap_values(input_row)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        shap_values_class1 = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values
        base_value = explainer.expected_value

    # Fix: some SHAP versions return expected_value as an array (e.g. one value
    # per class) instead of a single number. The waterfall plot needs one plain
    # number, so we pull out a single scalar here.
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    base_value = float(base_value)

    explanation = shap.Explanation(
        values=shap_values_class1[0],
        base_values=base_value,
        data=input_row.iloc[0],
        feature_names=feature_columns
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False, max_display=10)
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward 'will purchase'. "
        "Blue bars push it toward 'will not purchase'. "
        "The longer the bar, the bigger that feature's impact on this specific prediction."
    )