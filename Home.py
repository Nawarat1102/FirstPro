import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ====== ฟังก์ชันโหลด Lottie ======
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ====== เมนูหน้าเพจ ======
st.page_link("Home.py", label="หน้าแรก", icon="🏠")
st.page_link("pages/DTree.py", label="การทำนายข้อมูลด้วยเทคนิคต้นไม้ตัดสินใจ", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="การทำนายข้อมูลด้วยเทคนิค Naive Baye", icon="2️⃣")
st.page_link("pages/KnnwithHeart.py", label="การทำนายข้อมูลด้วยเทคนิค KNN", icon="3️⃣")
st.page_link("http://www.google.com", label="Google", icon="🌎")

# ====== โหลด Lottie ======
lottie_url_hello = "https://lottie.host/a6ba2cb1-4445-4e24-b0c0-29c910e30d35/siFuTj6Dck.json"
lottie_hello = load_lottieurl(lottie_url_hello)

# ====== Layout ======
col1, col2 = st.columns(2)
with col1:
    st.header("")
    st.image("./img/a1.png")

with col2:
    st.header("")
    st.image("./img/b8.jpg")

# ====== แสดง Animation ======
if lottie_hello:
    st_lottie(lottie_hello, height=250, key="hello")
