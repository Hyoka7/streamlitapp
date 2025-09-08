import gdown
import os
import streamlit as st
from gensim.models import KeyedVectors
import pandas as pd

@st.cache_resource
def load_w2v_model():
    file_id = "1M9yZ686yEUc1izurAemDeIp7j1D-sH-L"
    output = "jawiki.w2vmodel.bin"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return KeyedVectors.load_word2vec_format(output, binary=True)

@st.cache_resource
def load_fasttext_model():
    file_id = "1RPOF3c-mgFOCyk51oseFZGHRpjnAhD1e" 
    output = "jawiki.fasttextmodel.bin"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return KeyedVectors.load_word2vec_format(output, binary=True)

with st.spinner("モデルをロード中...(時間がかかる場合があります)"):
    w2v_model = load_w2v_model()
    fasttext_model = load_fasttext_model()

st.title("Word2Vec & FastText 類似単語検索")
word = st.text_input("単語を入力してください:")

if st.button("検索"):
    if not word:
        st.warning("単語を入力してください")
    else:
        col1, col2 = st.columns(2)
        with col1:
            try:
                res_w2v = w2v_model.most_similar(word)
                df_w2v = pd.DataFrame(res_w2v, columns=["単語", "類似度"])
                df_w2v.index = df_w2v.index + 1  # 1-indexed
                df_w2v["類似度"] = df_w2v["類似度"].map(lambda x: f"{x:.4f}")  # 小数点4桁
                st.markdown("### Word2Vec 類似単語")
                st.markdown(
                    df_w2v.style.set_properties(**{'text-align': 'center'})
                    .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                    .to_html(),
                    unsafe_allow_html=True
                )
            except KeyError:
                st.error(f"{word} は Word2Vec モデルに存在しません")
        with col2:
            try:
                res_ft = fasttext_model.most_similar(word)
                df_ft = pd.DataFrame(res_ft, columns=["単語", "類似度"])
                df_ft.index = df_ft.index + 1  # 1-indexed
                df_ft["類似度"] = df_ft["類似度"].map(lambda x: f"{x:.4f}")  # 小数点4桁
                st.markdown("### FastText 類似単語")
                st.markdown(
                    df_ft.style.set_properties(**{'text-align': 'center'})
                    .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                    .to_html(),
                    unsafe_allow_html=True
                )
            except KeyError:
                st.error(f"{word} は FastText モデルに存在しません")
