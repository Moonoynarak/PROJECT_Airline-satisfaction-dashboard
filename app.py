import streamlit as st
import pandas as pd
import pickle

# โหลดของที่เซฟไว้
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

st.title("✈ Airline Satisfaction Prediction")

# รับค่า input
age = st.slider("Age", 10, 80, 30)
distance = st.slider("Flight Distance", 100, 5000, 1000)
dep_delay = st.slider("Departure Delay (min)", 0, 300, 10)
arr_delay = st.slider("Arrival Delay (min)", 0, 300, 5)

customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

# สร้าง dataframe ก่อน encode
input_df = pd.DataFrame({
    'Age': [age],
    'Flight Distance': [distance],
    'Departure Delay in Minutes': [dep_delay],
    'Arrival Delay in Minutes': [arr_delay],
    'Customer Type': [customer_type],
    'Type of Travel': [travel_type],
    'Class': [travel_class]
})

# ทำ dummy เหมือนตอน train
input_df = pd.get_dummies(
    input_df,
    columns=['Customer Type', 'Type of Travel','Class'],
    drop_first=True
)

# เติมคอลัมน์ที่ขาด
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# เรียงคอลัมน์ให้ตรง
input_df = input_df[feature_columns]

# scale เฉพาะ num_cols
num_cols = [
    'Age',
    'Flight Distance',
    'Departure Delay in Minutes',
    'Arrival Delay in Minutes'
]

input_df[num_cols] = scaler.transform(input_df[num_cols])
st.write(input_df)
# predict
if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Prediction: Satisfied 😊")
    else:

        st.error("Prediction: Not Satisfied 😕")
