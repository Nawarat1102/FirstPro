from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# ====== ฟังก์ชันโหลด Lottie ======
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ====== โหลด Lottie ======
lottie_url_hello = "https://lottie.host/a6ba2cb1-4445-4e24-b0c0-29c910e30d35/siFuTj6Dck.json"
lottie_hello = load_lottieurl(lottie_url_hello)

st.title('การทำนายข้อมูลโรคเบาหวานระยะเริ่มต้น K-Nearest Neighbor')

# ----------------- Layout -----------------
col1, col2 = st.columns(2)
with col1:
   st.image("./img/b2.jpg")
with col2:
   st.image("./img/b4.jpg")

# ----------------- ข้อมูล -----------------
html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px;
            border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคเบาหวานสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

dt = pd.read_csv("./data/Diabetes.csv")
dt = dt.replace({
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0,
    'Positive': 1, 'Negative': 0
})

st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# ----------------- Visualization -----------------
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

st.write(f"### 🎯 Boxplot: {feature} แยกตามผลโรคเบาหวาน")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='class', y=feature, ax=ax)
st.pyplot(fig)

if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='class')
    st.pyplot(fig2)

# ----------------- ทำนาย -----------------
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;
            border-style:'solid';border-color:black">
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
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris-3.csv") 
   X = dt.drop('class', axis=1)
   y = dt['class']

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.success("⚠️ท่านมีความเสี่ยงเบาหวานระยะเริ่มต้น (｡ŏ﹏ŏ)")
    st.image("./img/b3.jpg")
   else:
    st.success("✅ท่านไม่มีความเสี่ยงเบาหวาน (≧▽≦)")
    st.image("./img/b5.jpg")
    if lottie_hello:
        st_lottie(lottie_hello, height=250, key="hello")

else:
    st.write("ไม่ทำนาย")