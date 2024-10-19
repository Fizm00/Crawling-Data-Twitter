import streamlit as st
import pandas as pd
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import time

# Load models and feature extraction
feature_bow = pickle.load(open("model/feature-bow.p", 'rb'))
model_nb = pickle.load(open('model/model-nb.p', 'rb'))
model_nn = pickle.load(open('model/model-nn.p', 'rb'))

# Load dataset for visualization
data = pd.read_csv('data/dataset_predicted_sentiment.csv')  # Dataset for clustering
document = data['Tweet'].tolist()

# Vectorize the document for clustering
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)

# Clustering model
true_k = 15
model_kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10, random_state=10)
model_kmeans.fit(X)

# PCA for visualization
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(X.toarray())
reduced_cluster_centers = pca.transform(model_kmeans.cluster_centers_)

# Function to clean text
def cleansing(string):
    string = string.lower()
    string = re.sub(r'http[s]?://\S+', '', string)  # Remove URLs
    string = re.sub(r'\s*@\w+\s*', ' ', string)
    string = re.sub(r'[^a-zA-Z0-9\s]', ' ', string)  # Remove non-alphanumeric characters
    string = re.sub(r'\brt\b', '', string)  # Remove 'rt'
    string = re.sub(r'\s+', ' ', string).strip()  # Remove extra spaces
    return string

# Predict sentiment function
def predict_sentiment(text):
    text = cleansing(text)
    text_feature = feature_bow.transform([text])
    sentiment_nb = model_nb.predict(text_feature)[0]
    sentiment_nn = model_nn.predict(text_feature)[0]
    return sentiment_nb, sentiment_nn

# Function to clean, train, and rename columns in uploaded CSV
def cleansing_with_stemming(string):
    string = string.lower()
    string = re.sub(r'http[s]?://\S+', '', string)  # Remove links
    string = re.sub(r'@\w+', '', string)  # Remove mentions
    string = re.sub(r'[^a-zA-Z0-9\s]', ' ', string)
    string = re.sub(r'\brt\b', '', string)
    string = re.sub(r'\s+', ' ', string).strip()
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    string = stemmer.stem(string)
    
    return string

# Modified process_csv with tqdm and progress bar
def process_csv(file):
    data = pd.read_csv(file)
    if 'full_text' in data.columns:
        data = data.rename(columns={'full_text': 'Tweet'})
        data['Tweet_clean'] = ''
        data['predicted_sentiment'] = ''
        
        # Show progress bar using st.progress and tqdm for progress visualization
        progress_bar = st.progress(0)
        for i, tweet in tqdm(enumerate(data['Tweet']), total=len(data['Tweet'])):
            cleaned_tweet = cleansing_with_stemming(tweet)
            sentiment_nb, _ = predict_sentiment(cleaned_tweet)
            data.at[i, 'Tweet_clean'] = cleaned_tweet
            data.at[i, 'predicted_sentiment'] = sentiment_nb

            # Update Streamlit progress bar
            progress_bar.progress((i + 1) / len(data))

            # Simulate processing time
            time.sleep(0.1)  # You can adjust or remove this if not necessary
        
        return data[['Tweet', 'Tweet_clean', 'predicted_sentiment']]
    else:
        return None

# Function to generate a word cloud
def generate_word_cloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def plot_clustering(reduced_features, reduced_cluster_centers, model_kmeans, X):
    """Visualisasi clustering tweet menggunakan PCA"""
    plt.figure(figsize=(10, 5))  # Ukuran lebih kecil
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model_kmeans.predict(X), cmap='rainbow', alpha=0.7)
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='black')
    plt.title("Visualisasi Clustering dengan PCA")
    st.pyplot(plt)

# Function to generate bar chart for most common words
def plot_most_common_words(text_data, num_words=10):
    words = ' '.join(text_data).split()
    most_common_words = Counter(words).most_common(num_words)
    
    words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Word', y='Frequency', data=words_df)
    plt.title(f'Top {num_words} Most Common Words')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Streamlit multi-page layout
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi Sentimen", "Visualisasi Data", "Upload CSV"])

# Page 1: Sentiment Prediction
if page == "Prediksi Sentimen":
    st.markdown(
        '<img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/c83c004e-1370-4756-88e5-4071de797088/dgdq8br-09cc7ad6-a021-47a5-b0e0-917b12b0f7a7.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2M4M2MwMDRlLTEzNzAtNDc1Ni04OGU1LTQwNzFkZTc5NzA4OFwvZGdkcThici0wOWNjN2FkNi1hMDIxLTQ3YTUtYjBlMC05MTdiMTJiMGY3YTcuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.tqRMtE-b2QiI2nnefNxSDMJvZCcYqFmq2ccg_Xfzqb8" width="800" height="380" alt="GIF Animation" style="display: block; margin-left: auto; margin-right: auto;" />',
        unsafe_allow_html=True
    )
    st.title("Prediksi Sentimen")
    user_input = st.text_area("Masukkan teks untuk analisis sentimen:")
    if st.button("Prediksi Sentimen"):
        if user_input:
            sentiment_nb, sentiment_nn = predict_sentiment(user_input)
            st.write(f"Sentimen Prediksi (Naive Bayes): **{sentiment_nb}**")
            st.write(f"Sentimen Prediksi (Neural Network): **{sentiment_nn}**")
        else:
            st.write("Silakan masukkan teks untuk analisis.")

# Page 2: Data Visualization
elif page == "Visualisasi Data":
    st.title("Visualisasi Clustering Tweet")

    st.write("Menampilkan Clustering tweet yang telah diproses:")
    if st.button("Tampilkan Clustering"):
        plot_clustering(reduced_features, reduced_cluster_centers, model_kmeans, X)
    
    # Word Cloud
    st.header("Word Cloud dari Tweet")
    if st.button("Tampilkan Word Cloud"):
        generate_word_cloud(document)

    # Most Common Words
    st.header("Most Common Words")
    if st.button("Tampilkan 10 Kata Paling Sering"):
        plot_most_common_words(document)

# Page 3: CSV Upload
elif page == "Upload CSV":
    st.title("Upload CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.write("File berhasil diupload, sedang memproses...")
        processed_data = process_csv(uploaded_file)
        if processed_data is not None:
            st.write("Data yang telah diproses:")
            st.dataframe(processed_data.head())
            processed_data.to_csv("processed_data.csv", index=False)
            st.download_button("Unduh CSV yang telah diproses", data=processed_data.to_csv(index=False), file_name="processed_data.csv", mime="text/csv")
        else:
            st.error("File CSV tidak sesuai format, kolom 'full_text' tidak ditemukan.")
