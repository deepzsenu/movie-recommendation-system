from flask import Flask, request, jsonify
import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle

app = Flask(__name__)

# Load the trained model
with open('models/recommendation_model.pkl', 'rb') as f:
    algo = pickle.load(f)

# Load movies data
movies = pd.read_csv('data/movies.csv')

# Load cosine similarity matrix
with open('models/cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Endpoint to get recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    movie_title = request.args.get('movie_title')
    recommendations = hybrid_recommendation(user_id, movie_title, algo, cosine_sim)
    return jsonify(recommendations)

# Function to get hybrid recommendations
def hybrid_recommendation(user_id, title, algo, cosine_sim):
    movie_idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    hybrid_scores = []
    for idx in movie_indices:
        movie_id = movies.iloc[idx]['movieId']
        pred = algo.predict(user_id, movie_id).est
        hybrid_scores.append((movies.iloc[idx]['title'], pred))
    
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    return hybrid_scores

if __name__ == '__main__':
    app.run()
