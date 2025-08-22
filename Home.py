import streamlit as st

st.page_link("Home.py", label="หน้าแรก", icon="🏠")
st.page_link("pages/DTree.py", label="การทำนายข้อมูลด้วยเทคนิคต้นไม้ตัดสินใจ", icon="⓵")
st.page_link("pages/NaiveBaye.py", label="การทำนายข้อมูลด้วยเทคนิค Naive Baye", icon="⓶", disabled=False)
st.page_link("pages/KnnwithHeart.py", label="การทำนายข้อมูลด้วยเทคนิค KNN", icon="⓷", disabled=False)
st.page_link("http://www.google.com", label="Google", icon="🌎")