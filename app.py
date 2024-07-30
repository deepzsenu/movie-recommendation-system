from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from surprise import SVD as SurpriseSVD, Dataset, Reader

app = Flask(__name__)
CORS(app)

# Load the trained SVD model
with open('models/recommendation_model.pkl', 'rb') as f:
    algo = pickle.load(f)

# Load the NearestNeighbors model
with open('models/nn_model.pkl', 'rb') as f:
    nn = pickle.load(f)

# Load the movie data
movies = pd.read_csv('data/movies.csv')

# Create a TF-IDF Vectorizer to transform the genres into a feature matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

def get_nearest_neighbors(movie_id, n_neighbors=10):
    movie_idx = movies[movies['movieId'] == movie_id].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[movie_idx], n_neighbors=n_neighbors)
    return distances, indices

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    movie_title = request.args.get('movie_title')

    # Find the movie ID based on the movie title
    if movie_title not in movies['title'].values:
        return jsonify({"error": "Movie title not found"}), 404
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]

    # Get nearest neighbors
    distances, indices = get_nearest_neighbors(movie_id)
    recommended_movies = []
    for i, idx in enumerate(indices[0]):
        recommended_movies.append({
            "title": movies.iloc[idx]['title'],
            "distance": distances[0][i]
        })

    # Hybrid recommendation
    hybrid_scores = []
    for idx in indices[0]:
        movie_id_nn = movies.iloc[idx]['movieId']
        pred = algo.predict(user_id, movie_id_nn).est
        hybrid_scores.append({
            "title": movies.iloc[idx]['title'],
            "score": pred
        })

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x['score'], reverse=True)

    return jsonify({
        "recommended_movies": recommended_movies,
        "hybrid_recommendations": hybrid_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
