import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
# Load dữ liệu
danhgia_df = pd.read_csv('data/Danh_gia.csv')
sanpham_df = pd.read_csv('data/San_pham.csv')

# Tiền xử lý dữ liệu
sanpham_df.dropna(inplace=True)
danhgia_df.dropna(inplace=True)

# Xử lý mô tả sản phẩm (giữ lại các cột cần thiết)
sanpham_df['content'] = sanpham_df['ten_san_pham'] + " " + sanpham_df['mo_ta']
sanpham_df = sanpham_df[['ma_san_pham', 'content']]

# Ma trận user-item
user_item_matrix = sanpham_df.pivot(index='user_id', columns='item_id', values='sanpham_df').fillna(0)
# Độ tương đồng giữa người dùng
user_similarity = cosine_similarity(user_item_matrix)

from sklearn.feature_extraction.text import TfidfVectorizer

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

# Gợi ý sản phẩm
def recommend_items(user_id, user_similarity, user_item_matrix, top_n=2):
    user_idx = user_id - 1
    similar_users = user_similarity[user_idx]
    scores = similar_users @ user_item_matrix.fillna(0).values
    item_ids = user_item_matrix.columns
    recommendations = pd.DataFrame({'item_id': item_ids, 'score': scores}).sort_values(by='score', ascending=False)
    return recommendations.head(top_n)


# Giao diện Streamlit
st.title("Hệ thống gợi ý sản phẩm")
option = st.selectbox("Chọn loại gợi ý", ("Content-Based", "Collaborative Filtering (Surprise)"))
input_id = st.text_input("Nhập User ID (CF) hoặc Product ID (Content-Based):")

if st.button("Gợi ý"):
    if option == "Content-Based":
         recommendations = recommend_content(product_id=int(input_id))
    elif option == "Collaborative Filtering (Surprise)":
        recommendations = recommend_items(1, user_similarity, user_item_matrix)

    if not recommendations.empty:
        st.write("Danh sách gợi ý:")
        st.write(recommendations)
    else:
        st.write("Không tìm thấy gợi ý!")
