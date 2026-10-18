import streamlit as st
import pandas as pd
import joblib

model = joblib.load('house_price_predition.pkl')
cols = joblib.load('column.pkl2')


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Boston House Price Predictor")
st.caption("Neighbourhood ki details bhar, model median house price (in $1000s) predict karega.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location & Area")
    CRIM = st.number_input("CRIM - Crime rate per capita", 0.0, 100.0, 0.25, step=0.01)
    ZN = st.number_input("ZN - Residential land zoned (%)", 0.0, 100.0, 0.0, step=0.5)
    INDUS = st.number_input("INDUS - Non-retail business acres (%)", 0.0, 30.0, 9.7, step=0.1)
    CHAS = st.selectbox("CHAS - Charles River ke paas?", [0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No")
    NOX = st.number_input("NOX - Nitric oxide concentration", 0.3, 1.0, 0.54, step=0.001, format="%.3f")

with col2:
    st.subheader("House Details")
    RM = st.number_input("RM - Avg rooms per dwelling", 3.0, 9.0, 6.2, step=0.1)
    AGE = st.number_input("AGE - Owner-occupied units built before 1940 (%)", 0.0, 100.0, 68.0, step=0.5)
    DIS = st.number_input("DIS - Distance to employment centres", 1.0, 13.0, 3.8, step=0.1)
    RAD = st.number_input("RAD - Highway accessibility index", 1, 24, 5, step=1)
    TAX = st.number_input("TAX - Property tax rate per $10,000", 150, 800, 330, step=1)

with col3:
    st.subheader("Society")
    PTRATIO = st.number_input("PTRATIO - Pupil-teacher ratio", 12.0, 22.0, 19.0, step=0.1)
    B = st.number_input("B - Black population index", 0.0, 400.0, 390.0, step=1.0)
    LSTAT = st.number_input("LSTAT - Lower status population (%)", 1.0, 40.0, 12.0, step=0.1)

if st.button("Predict Price", type="primary"):
    data = pd.DataFrame([{
        'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS, 'NOX': NOX,
        'RM': RM, 'AGE': AGE, 'DIS': DIS, 'RAD': RAD, 'TAX': TAX,
        'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT,
    }])
    data = data[cols]  # training wala column order

    price = model.predict(data)[0]
    st.success(f"Predicted Price: ${price * 1000:,.0f}  ({price:.2f} in $1000s)")