import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained SVM model and vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Analyze sentiment
def analyze_sentiment(text, model):
    if not text:
        return "Neutral"
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Create WordCloud
def create_wordcloud(data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='RdPu').generate(data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, pad=20)
    st.pyplot(plt)

# Plot sentiment distribution
def plot_sentiment_distribution(data, sentiment_column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=sentiment_column)
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Jumlah Review')
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="❤️", layout="centered")

    # Title
    st.title("✨ Analisis Kualitas Produk Makeup PinkFlash Berdasarkan Review ✨")
    st.markdown("<h5 style='text-align: center; color: pink;'>Analyze the sentiment of product reviews!</h5>", unsafe_allow_html=True)

    # Tabs for different features
    tab1, tab2 = st.tabs(["🔍 Manual Review Analysis", "📊 Dataset Analysis"])

    # Tab 1: Manual Review Analysis
    with tab1:
        st.subheader("🔍 Manual Review Analysis")
        review_text = st.text_area("Enter Product Review:", "", height=150)
        
        if st.button("Analyze Sentiment (Manual)"):
            sentiment = analyze_sentiment(review_text, svm_model)
            if sentiment == "Positive":
                st.success(f"Predicted Sentiment: *{sentiment}*")
            elif sentiment == "Negative":
                st.error(f"Predicted Sentiment: *{sentiment}*")
            else:
                st.warning(f"Predicted Sentiment: *{sentiment}*")
    
    # Tab 2: Dataset Analysis
    with tab2:
        st.subheader("📊 Dataset Analysis")
        uploaded_file = st.file_uploader("Upload file CSV", type="csv")

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.markdown("### 📊 Informasi Dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Review", len(data))
            with col2:
                st.metric("Rating Rata-rata", round(data['Rating'].mean(), 2))
            with col3:
                st.metric("% Recommended", f"{(data['Is Recommended'].mean() * 100):.1f}%")
            
            # Analyze sentiment on dataset
            st.subheader("🎯 Analisis Sentimen pada Dataset")
            if st.button("Jalankan Analisis Sentimen (Dataset)"):
                data['Sentiment'] = data['cleaned_review'].apply(lambda x: analyze_sentiment(x, svm_model))
                plot_sentiment_distribution(data, 'Sentiment', 'Distribusi Sentimen (Model: SVM)')

                # WordCloud for positive and negative sentiments
                st.markdown("### WordCloud")
                positive_reviews = " ".join(data[data['Sentiment'] == 'Positive']['cleaned_review'].dropna())
                create_wordcloud(positive_reviews, "Word Cloud - Sentimen Positif (Model: SVM)")

                negative_reviews = " ".join(data[data['Sentiment'] == 'Negative']['cleaned_review'].dropna())
                create_wordcloud(negative_reviews, "Word Cloud - Sentimen Negatif (Model: SVM)")
            
            # Download analyzed dataset
            st.subheader("📥 Download Hasil Analisis")
            if st.button("Simpan Hasil"):
                output_file = "hasil_analisis_sentimen.csv"
                data.to_csv(output_file, index=False)
                st.success(f"Hasil analisis telah disimpan sebagai {output_file}.")
                with open(output_file, 'rb') as file:
                    st.download_button("Download Hasil Analisis", file, file_name=output_file, mime='text/csv')

if __name__ == "__main__":
    main()