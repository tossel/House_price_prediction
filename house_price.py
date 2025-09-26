import pickle
import numpy as np
import streamlit as st

# ✅ Load the trained XGBoost model

 
with open("xgb.pkl", "rb") as pickle_in:
    model = pickle.load(pickle_in)

def main():
    st.title("🏠 House Price Prediction App")
    st.write("Enter the property details below to predict the house price.")

    # 🔹 Input fields
    col1, col2 = st.columns(2)
    Area = col1.number_input("Area (sq ft)", min_value=0)
    Bedrooms = col2.number_input("Number of Bedrooms", min_value=0)
    Bathrooms =col2.number_input("Numbers of Bathrooms", min_value=0)
    Floors = col1.number_input("Number of Floors", min_value=0)
    YearBuilt = col2.number_input("Year Built", min_value=1800, max_value=2025, step=1)
    Location = col1.number_input("Location Code", min_value=0)
    Condition = col2.number_input("Condition Rating", min_value=0.0, step=0.1)
    Garage = col1.number_input("Garage Size", min_value=0.0, step=0.1)
   

    # 🔹 Predict Button
    if st.button("Predict"):
        # Prepare input as 2D array for model
        input_values = np.array([[Area, Bedrooms, Bathrooms, Floors,
                                  YearBuilt, Location, Condition,
                                  Garage]])

        # Make prediction
        prediction = model.predict(input_values)
        predicted_price = prediction[0]

        # Display result
        st.success(f"💵 **Predicted House Price:** ${predicted_price:,.2f}")

if __name__ == '__main__':
    main()
    
    
    
    