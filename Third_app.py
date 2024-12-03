import joblib
import pandas as pd
import streamlit as st


# Load the trained model and input columns
try:
    Model = joblib.load("Third_Group.pkl")
    Inputs = joblib.load("Inputs.pkl")
except FileNotFoundError:
    st.error("Model or input files not found. Ensure 'Third_Group.pkl' and 'Inputs.pkl' are in the directory.")
    st.stop()

def prediction(data):
    return Model.predict(data)[0]

def Main():
    st.title("California Housing Price Prediction")
    st.write("Provide the following details to predict the median house value:")

    # User input fields
    longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31, value=-118.0, step=0.01)
    latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95, value=34.0, step=0.01)
    housing_median_age = st.slider("Housing Median Age", min_value=1, max_value=52, value=20)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=40000, value=1000)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=5000, value=200)
    population = st.number_input("Population", min_value=1, max_value=40000, value=1500)
    households = st.number_input("Households", min_value=1, max_value=5000, value=300)
    median_income = st.number_input("Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=3.0)

   
    ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    

    # Create DataFrame with all features
    data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity':  ocean_proximity
    }])

    # Check if DataFrame matches model input
    if not all(col in Inputs for col in data.columns):
        st.error("Mismatch between input columns and model expected columns.")
        return

    if st.button("Predict"):
        try:
            result = prediction(data)
            st.success(f"Predicted Median House Value: ${result:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    Main()
