import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.title('การทำนายความเสี่ยงเบาหวานระยะเริ่มต้นด้วย Naive Bayes')

# โหลดข้อมูล
dt = pd.read_csv("./data/Diabetes.csv")
# แปลงค่าหมวดหมู่เป็นตัวเลข
dt = dt.replace({
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0,
    'Positive': 1, 'Negative': 0
})

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='class')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

st.subheader("กรุณาใส่ข้อมูลเพื่อทำนายความเสี่ยงเบาหวาน")
A1 = st.number_input("อายุ")
A2 = st.number_input("เพศ (ชาย=1 หญิง=0)")
A3 = st.number_input("ปัสสาวะบ่อย (ใช่=1 ไม่ใช่=0)")
A4 = st.number_input("กระหายน้ำ (ใช่=1 ไม่ใช่=0)")
A5 = st.number_input("น้ำหนักลดลงอย่างรวดเร็ว (ใช่=1 ไม่ใช่=0)")
A6 = st.number_input("อ่อนแรง (ใช่=1 ไม่ใช่=0)")
A7 = st.number_input("หิวบ่อย (ใช่=1 ไม่ใช่=0)")
A8 = st.number_input("เชื้อราในอวัยวะเพศ (ใช่=1 ไม่ใช่=0)")
A9 = st.number_input("สายตาพร่ามัว (ใช่=1 ไม่ใช่=0)")
A10 = st.number_input("คัน (ใช่=1 ไม่ใช่=0)")
A11 = st.number_input("หงุดหงิดง่าย (ใช่=1 ไม่ใช่=0)")
A12 = st.number_input("แผลหายช้า (ใช่=1 ไม่ใช่=0)")
A13 = st.number_input("กล้ามเนื้ออ่อนแรงบางส่วน (ใช่=1 ไม่ใช่=0)")
A14 = st.number_input("กล้ามเนื้อตึง (ใช่=1 ไม่ใช่=0)")
A15 = st.number_input("ผมร่วง (ใช่=1 ไม่ใช่=0)")
A16 = st.number_input("โรคอ้วน (ใช่=1 ไม่ใช่=0)")



if st.button("ทำนายผล"):
    X = dt.drop('class', axis=1)
    y = dt['class']

    # สร้างโมเดล Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X, y)

    x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16]])
    out = nb_model.predict(x_input)

    st.write(out)  # แสดงผล raw (0 หรือ 1)

    if out[0] == 1:
        st.success("ท่านมีความเสี่ยงเบาหวานระยะเริ่มต้น")
        st.image("./img/b3.jpg")
    else:
        st.success("ท่านไม่มีความเสี่ยงเบาหวาน")
        st.image("./img/b5.jpg")
else:
    st.write("ไม่ทำนาย")
