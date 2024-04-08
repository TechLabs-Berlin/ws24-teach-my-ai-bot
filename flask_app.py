from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings

app = Flask(__name__)


warnings.filterwarnings("ignore", category=UserWarning)

# TfidfVectorizer instance from the saved pickle file
pickle_file_path = os.path.join(os.path.expanduser("~"), "Desktop", "Recommender Model", "vectorizer.pkl")
with open(pickle_file_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)


df = pd.read_csv(r'C:\Users\Bibin Karthikeyan\Desktop\data\coursera_udemy.csv')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


preprocessed_summaries = df['Summary'].apply(preprocess_text)


tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_summaries)

def recommend_courses(pdf_text):
    preprocessed_pdf_text = preprocess_text(pdf_text)
    pdf_tfidf = tfidf_vectorizer.transform([preprocessed_pdf_text])
    similarities = cosine_similarity(pdf_tfidf, tfidf_matrix)
    top_indices = similarities.argsort(axis=1)[:, -5:][:, ::-1]
    top_courses = df.iloc[top_indices.flatten()]
    return top_courses

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    pdf_text = data['pdf_text']
    print("Received PDF Text:", pdf_text)
    top_courses = recommend_courses(pdf_text)

    recommendation_data = {
        "courses": top_courses.to_dict(orient='records')
    }
    print("Recommendation Data:", recommendation_data)
    return jsonify(recommendation_data)

if __name__ == '__main__':
    app.run(debug=True)
