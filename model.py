import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
from surprise import SVD, Dataset, Reader

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

# Save the preprocessed dataset to a CSV file
data.to_csv('data/movies.csv', index=False)


# Load the merged data
data = pd.read_csv('data/movies.csv')

# Load data into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train the SVD algorithm
algo = SVD()
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

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save the cosine similarity matrix
with open('models/cosine_sim.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)
