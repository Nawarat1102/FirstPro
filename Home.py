import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ====== ฟังก์ชันโหลด Lottie ======
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

import json
from streamlit_lottie import st_lottie

with open("./img/Health.json") as f:
    lottie_hm = json.load(f)

st.page_link("Home.py", label="หน้าแรก", icon="🏠")
st.page_link("pages/DTree.py", label="การทำนายข้อมูลด้วยเทคนิคต้นไม้ตัดสินใจ", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="การทำนายข้อมูลด้วยเทคนิค Naive Baye", icon="2️⃣")
st.page_link("pages/KnnwithHeart.py", label="การทำนายข้อมูลด้วยเทคนิค KNN", icon="3️⃣")
st.page_link("http://www.google.com", label="Google", icon="🌎")


# ====== แสดง Animation ======
st_lottie(lottie_hm, height=200)

col1, col2 = st.columns(2)
with col1:
    st.header("")
    st.image("./img/a1.png")

with col2:
    st.header("")
    st.image("./img/b8.jpg")


