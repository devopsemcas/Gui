import pandas as pd
import pickle
import streamlit as st
from surprise import Dataset, Reader, SVD

# Load dữ liệu
danhgia_df = pd.read_csv('data/Danh_gia.csv')
sanpham_df = pd.read_csv('data/San_pham.csv')

# Tiền xử lý dữ liệu
sanpham_df.dropna(inplace=True)
danhgia_df.dropna(inplace=True)

# Xử lý mô tả sản phẩm (giữ lại các cột cần thiết)
sanpham_df['content'] = sanpham_df['ten_san_pham'] + " " + sanpham_df['mo_ta']
sanpham_df = sanpham_df[['ma_san_pham', 'content']]

# Chuẩn bị dữ liệu cho Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(danhgia_df[['ma_khach_hang', 'ma_san_pham', 'so_sao']], reader)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # Tạo TF-IDF và ma trận cosine
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# tfidf_matrix = tfidf_vectorizer.fit_transform(sanpham_df['content'])
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Lưu TF-IDF và ma trận cosine
# with open('tfidf_vectorizer.pkl', 'wb') as f:
#     pickle.dump(tfidf_vectorizer, f)

# with open('cosine_sim.pkl', 'wb') as f:
#     pickle.dump(cosine_sim, f)

# Tải TF-IDF vectorizer và ma trận cosine
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# # Huấn luyện mô hình Collaborative Filtering với SVD
# algorithm = SVD()
# trainset = data.build_full_trainset()
# algorithm.fit(trainset)

# # Lưu mô hình Surprise
# with open('cf_model.pkl', 'wb') as f:
#     pickle.dump(algorithm, f)

# Tải mô hình
with open('cf_model.pkl', 'rb') as f:
    algorithm = pickle.load(f)

def recommend_content(product_id, top_n=5):
    try:
        # Lấy index của sản phẩm
        idx = sanpham_df.index[sanpham_df['ma_san_pham'] == product_id].tolist()[0]

        # Lấy danh sách tương đồng
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        product_indices = [i[0] for i in sim_scores]

        # Trả về các sản phẩm gợi ý
        return sanpham_df.iloc[product_indices][['ma_san_pham', 'content']]
    except IndexError:
        return pd.DataFrame({'ma_san_pham': [], 'content': []})


# Hàm gợi ý Collaborative Filtering
def recommend_cf(user_id, top_n=5):
    try:
        all_products = danhgia_df['ma_san_pham'].unique()
        scores = [(product, algorithm.predict(user_id, product).est) for product in all_products]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        product_ids = [s[0] for s in scores]
        return sanpham_df[sanpham_df['ma_san_pham'].isin(product_ids)]
    except:
        return pd.DataFrame()

# Giao diện Streamlit
st.title("Hệ thống gợi ý sản phẩm")
option = st.selectbox("Chọn loại gợi ý", ("Content-Based", "Collaborative Filtering (Surprise)"))
input_id = st.text_input("Nhập User ID (CF) hoặc Product ID (Content-Based):")

if st.button("Gợi ý"):
    if option == "Content-Based":
         recommendations = recommend_content(product_id=int(input_id))
    elif option == "Collaborative Filtering (Surprise)":
        recommendations = recommend_cf(user_id=int(input_id))

    if not recommendations.empty:
        st.write("Danh sách gợi ý:")
        st.write(recommendations)
    else:
        st.write("Không tìm thấy gợi ý!")
