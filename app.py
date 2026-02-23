import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Section 1 - Dataset Overview
# โหลดข้อมูล
df = pd.read_csv("Airline_customer_satisfaction.csv")

st.title("✈ Airline Satisfaction Dashboard")

st.header("🔹Dataset Overview")

col1, col2 = st.columns(2)

# จำนวนข้อมูล
col1.metric("Total Assessors", len(df))

# Pie Chart
satisfaction_counts = df["satisfaction"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    satisfaction_counts,
    labels=satisfaction_counts.index,
    autopct='%1.1f%%',
    colors=plt.cm.Pastel2.colors
)
ax.set_title("Satisfaction Distribution")

col2.pyplot(fig)

st.subheader("Average Arrival Delay by Satisfaction")
delay_avg = df.groupby("satisfaction")["Arrival Delay in Minutes"].mean()
st.bar_chart(delay_avg)

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    corr,
    annot=True, 
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)

st.divider()
# Section 2
st.header("🔹Interactive Filter")
st.subheader("Bar Chart of Satisfaction")

selected_class = st.selectbox(
    "Select Class",
    df["Class"].unique()
)

max_delay = st.slider(
    "Maximum Arrival Delay (minutes)",
    0, 300, 60
)

filtered_df = df[
    (df["Class"] == selected_class) &
    (df["Arrival Delay in Minutes"] <= max_delay)
]

st.write("Filtered Data Count:", len(filtered_df))
st.bar_chart(filtered_df["satisfaction"].value_counts())
st.divider()

st.subheader("Distribution of Data by Satisfaction")
feature_columns = pickle.load(open("features.pkl", "rb"))
dat_col = feature_columns.copy()
ctr = [
    "Customer Type_disloyal Customer",
    "Type of Travel_Personal Travel",
    "Class_Eco",
    "Class_Eco Plus"
]
dat_col = [
    col for col in dat_col
    if col not in ctr
]
selected_col = st.selectbox(
    "Select Data",
    dat_col
)
fig, ax = plt.subplots()

sns.boxplot(
    x='satisfaction',
    y=selected_col,
    data=df,
    ax=ax
)

st.pyplot(fig)

# โหลดของที่เซฟไว้
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title("🔹 Airline Satisfaction Prediction")

# รับค่า input
age = st.slider("Age", 10, 80, 30)
distance = st.slider("Flight Distance (km)", 100, 5000, 1000)
dep_delay = st.slider("Departure Delay (min)", 0, 300, 10)
arr_delay = st.slider("Arrival Delay (min)", 0, 300, 5)
seat_c = st.slider("Seat Comfort (Score)", 0, 5, 5)
inf_e = st.slider("Inflight Entertainment (Score)", 0, 5, 5)
eo_ob = st.slider("Ease of Online booking (Score)", 0, 5, 5)
on_sup = st.slider("Online Support (Score)", 0, 5, 5)
ob_ser = st.slider("On-board Service (Score)", 0, 5, 5)
f_d = st.slider("Food and Drink (Score)", 0, 5, 5)
lr_ser = st.slider("Leg Room Service (Score)", 0, 5, 5)

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
    'Class': [travel_class],
    'Seat comfort' : [seat_c],
    'Inflight entertainment' : [inf_e],
    'Ease of Online booking' : [eo_ob],
    'Online support' : [on_sup],
    'On-board service' : [ob_ser],
    'Food and drink' : [f_d],
    'Leg room service' : [lr_ser]
})

# ทำ dummy เหมือนตอน train
if input_df.loc[0, 'Customer Type'] == "Loyal Customer" :
    input_df['Customer Type_disloyal Customer'] = 0
else :
    input_df['Customer Type_disloyal Customer'] = 1
    
if input_df.loc[0, 'Type of Travel'] == "Business travel" :
    input_df['Type of Travel_Personal Travel'] = 0
else :
    input_df['Type of Travel_Personal Travel'] = 1

if input_df.loc[0, 'Class'] == "Business" :
    input_df['Class_Eco'] = 0
    input_df['Class_Eco Plus'] = 0
elif input_df.loc[0, 'Class'] == "Eco" :
    input_df['Class_Eco'] = 1
    input_df['Class_Eco Plus'] = 0
else :
    input_df['Class_Eco'] = 0
    input_df['Class_Eco Plus'] = 1


# เติมคอลัมน์ที่ขาด
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 4

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
#st.write(input_df)
#st.write(type(model))
#st.write(model)
# predict
if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Prediction: Satisfied 😊")
    else:

        st.error("Prediction: Not Satisfied 😕")

st.divider()
st.subheader("Top 10 Feature Importance")

importances = model.feature_importances_
feature_names = feature_columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
top10 = importance_df.head(10)
st.bar_chart(top10.set_index("Feature"))


























