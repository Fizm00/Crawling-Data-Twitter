import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load models and feature extraction
feature_bow = pickle.load(open("model/feature-bow.p", 'rb'))
model_nb = pickle.load(open('model/model-nb.p', 'rb'))
model_nn = pickle.load(open('model/model-nn.p', 'rb'))

# Load the dataset
data = pd.read_csv('data/dataset_predicted_sentiment.csv')  # Load dataset
document = data['Tweet'].tolist()

# Vectorize the document
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

# Streamlit layout
st.title("Dashboard Sentiment Analysis dan Clustering")

# User input for sentiment prediction
st.header("Prediksi Sentimen")
user_input = st.text_area("Masukkan teks untuk analisis sentimen:")
if st.button("Prediksi Sentimen"):
    if user_input:
        sentiment_nb, sentiment_nn = predict_sentiment(user_input)
        st.write(f"Sentimen Prediksi (Naive Bayes): **{sentiment_nb}**")
        st.write(f"Sentimen Prediksi (Neural Network): **{sentiment_nn}**")
    else:
        st.write("Silakan masukkan teks untuk analisis.")

# Display clustering
st.header("Hasil Clustering Tweet")

if st.button("Tampilkan Clustering"):
    # Scatter plot for cluster visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model_kmeans.predict(X), cmap='rainbow', alpha=0.7)
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='black')
    plt.title("Visualisasi Clustering dengan PCA")
    st.pyplot(plt)

# Display the clustered data
st.header("Data Tweet yang Sudah Dikelompokkan")
st.dataframe(data[['Tweet', 'Tweet_clean', 'predicted_sentiment']])
