import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.title("Decision Tree สำหรับทำนายความเสี่ยงเบาหวานระยะเริ่มต้น")

# โหลดข้อมูล
dt = pd.read_csv("./data/Diabetes.csv")

# แปลงค่า Yes/No, Male/Female, Positive/Negative เป็น 1/0
dt = dt.replace({
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0,
    'Positive': 1, 'Negative': 0
})

# แยก Features / Target
X = dt.drop('class', axis=1)
y = dt['class']

# สร้างโมเดล Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)




st.subheader("กรุณาใส่ข้อมูลเพื่อทำนายความเสี่ยงเบาหวาน")
A1 = st.number_input("อายุ")
A2 = st.number_input("เพศ (ชาย=1 หญิง=0)")
A3 = st.number_input("ปัสสาวะบ่อย (ใช่=1 ไม่ใช่=2)")
A4 = st.number_input("กระหายน้ำ (ใช่=1 ไม่ใช่=2)")
A5 = st.number_input("น้ำหนักลดลงอย่างรวดเร็ว (ใช่=1 ไม่ใช่=2)")
A6 = st.number_input("อ่อนแรง (ใช่=1 ไม่ใช่=2)")
A7 = st.number_input("หิวบ่อย (ใช่=1 ไม่ใช่=2)")
A8 = st.number_input("เชื้อราในอวัยวะเพศ (ใช่=1 ไม่ใช่=2)")
A9 = st.number_input("สายตาพร่ามัว (ใช่=1 ไม่ใช่=2)")
A10 = st.number_input("คัน (ใช่=1 ไม่ใช่=2)")
A11 = st.number_input("หงุดหงิดง่าย (ใช่=1 ไม่ใช่=2)")
A12 = st.number_input("แผลหายช้า (ใช่=1 ไม่ใช่=2)")
A13 = st.number_input("กล้ามเนื้ออ่อนแรงบางส่วน (ใช่=1 ไม่ใช่=2)")
A14 = st.number_input("กล้ามเนื้อตึง (ใช่=1 ไม่ใช่=2)")
A15 = st.number_input("ผมร่วง (ใช่=1 ไม่ใช่=2)")
A16 = st.number_input("โรคอ้วน (ใช่=1 ไม่ใช่=2)")




if st.button("ทำนายผล"):
    x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16]])
    out = dt_model.predict(x_input)

    st.write(out)  

    if out[0] == 1:
        st.success("ท่านมีความเสี่ยงเบาหวานระยะเริ่มต้น")
        st.image("./img/b3.jpg")
    else:
        st.success("ท่านไม่มีความเสี่ยงเบาหวาน")
        st.image("./img/b5.jpg")
else:
    st.write("ไม่ทำนาย")
