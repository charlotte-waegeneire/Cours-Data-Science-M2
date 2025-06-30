# streamlit_app_enhanced.py
import streamlit as st
import requests
import json
import pandas as pd

# Configure the page
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor", page_icon="🚢", layout="wide"
)

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000"


def main():
    st.title("🚢 Titanic Survival Predictor")
    st.write(
        "Enter passenger details to predict survival probability using our trained ML model"
    )

    # Check API connection
    try:
        response = requests.get(f"{FASTAPI_URL}/")
        if response.status_code == 200:
            st.success("✅ Connected to API")
        else:
            st.warning("⚠️ API connection issues")
    except:
        st.error("❌ Cannot connect to API. Make sure FastAPI is running!")
        return

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "ℹ️ Model Info"])

    with tab1:
        single_prediction_tab()

    with tab2:
        model_info_tab()


def single_prediction_tab():
    st.header("Single Passenger Prediction")

    # Create input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            help="1 = First Class, 2 = Second Class, 3 = Third Class",
        )
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)

    with col2:
        fare = st.slider("Fare ($)", 0.0, 500.0, 32.0)
        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"],
            help="S = Southampton, C = Cherbourg, Q = Queenstown",
        )

    with col3:
        family_size = st.slider(
            "Family Size",
            0,
            10,
            0,
            help="Number of siblings/spouses + parents/children aboard",
        )

        # Show family size interpretation
        if family_size == 0:
            st.info("👤 Traveling alone")
        elif family_size <= 2:
            st.info("👥 Small family")
        elif family_size <= 4:
            st.info("👨‍👩‍👧‍👦 Medium family")
        else:
            st.info("👨‍👩‍👧‍👦‍👧‍👦 Large family")

    # Create prediction button
    if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
        # Prepare data for API call
        passenger_data = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Embarked": embarked,
            "FamilySize": family_size,
        }

        try:
            # Make API call to FastAPI
            with st.spinner("Making prediction..."):
                response = requests.post(
                    f"{FASTAPI_URL}/predict",
                    json=passenger_data,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                survival_prob = result.get("survival_probability", prediction)

                # Display results
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.success("🎉 **SURVIVED!**")
                        st.balloons()
                    else:
                        st.error("💀 **DID NOT SURVIVE**")

                with col2:
                    # Show probability if available
                    if isinstance(survival_prob, float):
                        st.metric(
                            "Survival Probability",
                            f"{survival_prob:.1%}",
                            delta=f"{survival_prob - 0.5:.1%}"
                            if survival_prob != prediction
                            else None,
                        )
                        st.progress(survival_prob)
                    else:
                        st.metric("Prediction Confidence", "Binary Classification")

                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.json(passenger_data)

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the API. Make sure FastAPI is running on http://localhost:8000"
            )
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


def model_info_tab():
    st.header("Model Information")

    try:
        response = requests.get(f"{FASTAPI_URL}/model_info")
        if response.status_code == 200:
            model_info = response.json()

            st.write("### Model Details:")
            st.json(model_info)

            if "features" in model_info and isinstance(model_info["features"], list):
                st.write("### Model Features:")
                for i, feature in enumerate(model_info["features"], 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("Could not retrieve model information")

    except Exception as e:
        st.error(f"Error getting model info: {str(e)}")

    # Setup instructions
    with st.expander("🔧 Setup Instructions"):
        st.markdown("""
        **To run this application:**
        
        1. **Start FastAPI server:**
        ```bash
        uvicorn main:app --reload
        ```
        
        2. **Run Streamlit app:**
        ```bash
        streamlit run streamlit_app_enhanced.py
        ```
        
        3. **Install dependencies:**
        ```bash
        pip install fastapi uvicorn streamlit requests pandas
        ```
        """)


if __name__ == "__main__":
    main()
