

import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# Tải mô hình
model_cf = joblib.load('model_cf.pkl')
tfidf = joblib.load('tfidf_model.pkl')
cosine_sim = joblib.load('cosine_sim.pkl')
products = joblib.load('products.pkl')

# Gợi ý dựa trên Collaborative Filtering
def recommend_cf(user_id, model, num_recommendations=5):
    items = list(products['ma_san_pham'])
    predictions = [(item, model.predict(user_id, item).est) for item in items]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    return predictions[:num_recommendations]

# Gợi ý dựa trên Content-Based Filtering
def recommend_content(product_id, num_recommendations=5):
    idx = products.index[products['ma_san_pham'] == product_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    product_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    return products.iloc[product_indices]

# Giao diện Streamlit
st.title("Hệ thống gợi ý sản phẩm")
st.sidebar.title("Chọn kiểu gợi ý")

option = st.sidebar.selectbox("Phương pháp:", ["Collaborative Filtering", "Content-Based Filtering"])

if option == "Collaborative Filtering":
    st.header("Collaborative Filtering")
    user_id = st.number_input("Nhập ID khách hàng:", min_value=1, step=1)
    if st.button("Gợi ý sản phẩm"):
        recommendations = recommend_cf(user_id, model_cf)
        for product, score in recommendations:
            st.write(f"Sản phẩm: {product}, Điểm dự đoán: {score:.2f}")

elif option == "Content-Based Filtering":
    st.header("Content-Based Filtering")
    product_id = st.number_input("Nhập mã sản phẩm:", min_value=1, step=1)
    if st.button("Gợi ý sản phẩm tương tự"):
        recommendations = recommend_content(product_id)
        st.write(recommendations)