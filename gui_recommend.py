import pandas as pd
import pickle
import streamlit as st
from gensim import corpora, models, similarities
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

# Xử lý dữ liệu với Gensim
sanpham_df['tokens'] = sanpham_df['content'].apply(lambda x: x.split())

# Tạo từ điển và ma trận TF-IDF
dictionary = corpora.Dictionary(sanpham_df['tokens'])
corpus = [dictionary.doc2bow(text) for text in sanpham_df['tokens']]
tfidf_model = models.TfidfModel(corpus)
index = similarities.MatrixSimilarity(tfidf_model[corpus])

# Lưu mô hình Gensim
with open('gensim_dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
with open('gensim_tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf_model, f)
with open('gensim_index.pkl', 'wb') as f:
    pickle.dump(index, f)

# Huấn luyện mô hình Collaborative Filtering với SVD
algorithm = SVD()
trainset = data.build_full_trainset()
algorithm.fit(trainset)

# Lưu mô hình Surprise
with open('cf_model.pkl', 'wb') as f:
    pickle.dump(algorithm, f)

# Tải mô hình
with open('gensim_dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
with open('gensim_tfidf.pkl', 'rb') as f:
    tfidf_model = pickle.load(f)
with open('gensim_index.pkl', 'rb') as f:
    index = pickle.load(f)
with open('cf_model.pkl', 'rb') as f:
    algorithm = pickle.load(f)

# Hàm gợi ý Content-Based với Gensim
def recommend_content_gensim(product_id, top_n=5):
    try:
        # Lấy index của sản phẩm cần gợi ý
        product_idx = sanpham_df[sanpham_df['ma_san_pham'] == product_id].index[0]
        query_bow = dictionary.doc2bow(sanpham_df.iloc[product_idx]['tokens'])
        sims = index[tfidf_model[query_bow]]
        sims = sorted(list(enumerate(sims)), key=lambda x: -x[1])[1:top_n+1]

        # Lấy sản phẩm tương tự
        recommendations = pd.DataFrame([{
            'ma_san_pham': sanpham_df.iloc[i]['ma_san_pham'],
            'content': sanpham_df.iloc[i]['content'],
            'similarity_score': sim_score
        } for i, sim_score in sims])
        return recommendations
    except IndexError:
        return pd.DataFrame()

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
option = st.selectbox("Chọn loại gợi ý", ("Content-Based (Gensim)", "Collaborative Filtering (Surprise)"))
input_id = st.text_input("Nhập User ID (CF) hoặc Product ID (Content-Based):")

if st.button("Gợi ý"):
    if option == "Content-Based (Gensim)":
        recommendations = recommend_content_gensim(product_id=int(input_id))
    elif option == "Collaborative Filtering (Surprise)":
        recommendations = recommend_cf(user_id=int(input_id))

    if not recommendations.empty:
        st.write("Danh sách gợi ý:")
        st.write(recommendations)
    else:
        st.write("Không tìm thấy gợi ý!")