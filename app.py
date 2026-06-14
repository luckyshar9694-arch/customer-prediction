import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model
from pathlib import Path

st.set_page_config(page_title="Customer Prediction System", layout="wide")


logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=120)


st.title("🚀 Customer Prediction System")
st.write("Churn Prediction | Lead Conversion | Machine Learning")


mode = st.selectbox("Select Prediction Type", [
    "Customer Churn Prediction",
    "Lead Conversion Prediction"
])


uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        model, accuracy, cm, encoders = train_model(df, target)

        st.success("Model Trained Successfully!")

        # SAVE
        st.session_state["model"] = model
        st.session_state["encoders"] = encoders
        st.session_state["data"] = df
        st.session_state["target"] = target

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")

        with col2:
            st.write("Confusion Matrix")
            st.write(cm)

        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig)


if "model" in st.session_state:
    st.subheader("🔮 Predict Customer Outcome")

    model = st.session_state["model"]
    encoders = st.session_state["encoders"]
    df = st.session_state["data"]
    target = st.session_state["target"]

    input_data = {}

    for col in df.columns:
        if col != target:

            # CATEGORICAL → dropdown
            if col in encoders:
                input_data[col] = st.selectbox(
                    f"{col}",
                    options=encoders[col].classes_
                )

            # NUMERIC → number
            else:
                input_data[col] = st.number_input(f"{col}", value=0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # APPLY SAME ENCODING
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        # RESULT
        if mode == "Customer Churn Prediction":
            if prediction == 1:
                st.error(f"⚠️ Customer will LEAVE ({prob*100:.2f}%)")
            else:
                st.success(f"✅ Customer will STAY ({prob*100:.2f}%)")

        elif mode == "Lead Conversion Prediction":
            if prediction == 1:
                st.success(f"💰 Customer WILL BUY ({prob*100:.2f}%)")
            else:
                st.warning(f"❌ Customer will NOT BUY ({prob*100:.2f}%)")


st.markdown("""
<hr>
<p style='text-align:center'>
<b>Lucky Sharma</b><br>
© 2026 Lucky Sharma. All rights reserved.<br>
Customer Prediction System | ML • Python • Scikit-learn
</p>
""", unsafe_allow_html=True)