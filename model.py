import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from surprise import SVD as SurpriseSVD, Dataset, Reader

# Load the ratings data
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Load the movies data
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=[
    'movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 
    'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Keep only relevant columns
movies = movies[['movieId', 'title', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]

# Combine genre columns into a single 'genres' column
genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies['genres'] = movies[genre_columns].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)

# Merge ratings and movies data
data = pd.merge(ratings, movies[['movieId', 'title', 'genres']], on='movieId')

# Limit to top 500 movies based on the number of ratings
top_movies = data['movieId'].value_counts().head(500).index
data = data[data['movieId'].isin(top_movies)]

# Save the preprocessed dataset to a CSV file
data.to_csv('data/movies.csv', index=False)

# Load the merged data
data = pd.read_csv('data/movies.csv')

# Load data into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train the SVD algorithm
algo = SurpriseSVD()
algo.fit(trainset)

# Save the model to a file
with open('models/recommendation_model.pkl', 'wb') as f:
    pickle.dump(algo, f)

# Load the movie data
movies = pd.read_csv('data/movies.csv')

# Preprocess the movie genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Create a TF-IDF Vectorizer to transform the genres into a feature matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Use NearestNeighbors to find the nearest neighbors for each movie
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# Save the NearestNeighbors model
with open('models/nn_model.pkl', 'wb') as f:
    pickle.dump(nn, f)

# Function to get nearest neighbors
def get_nearest_neighbors(movie_id, n_neighbors=10):
    movie_idx = movies[movies['movieId'] == movie_id].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[movie_idx], n_neighbors=n_neighbors)
    return distances, indices

# Example usage
movie_id = 1  # Toy Story (1995)
distances, indices = get_nearest_neighbors(movie_id)
for i, idx in enumerate(indices[0]):
    print(f"Movie: {movies.iloc[idx]['title']}, Distance: {distances[0][i]}")
