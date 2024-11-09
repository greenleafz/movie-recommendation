from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the secret key from an environment variable or set a default
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')

# Adjust DATABASE_URL to be compatible with SQLAlchemy
uri = os.environ.get("DATABASE_URL", "sqlite:///local.db")  # Get the database URL from environment
if uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure session to use the database
app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Initialize extensions
db = SQLAlchemy(app)
app.config['SESSION_SQLALCHEMY'] = db

# Set the session table name (optional)
app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'

Session(app)

# Create the database tables
with app.app_context():
    db.create_all()

# Load the SVD matrix and movie titles for SVD recommendations
with open('./app/models/svd_matrix.pkl', 'rb') as f:
    svd_matrix = pickle.load(f)

with open('./app/models/movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

# Convert movie titles to a list if not already
if isinstance(movie_titles, np.ndarray):
    movie_titles = movie_titles.tolist()

# Create a mapping from movie titles to indices
movie_indices = {title: idx for idx, title in enumerate(movie_titles)}

# Load the NCF model and necessary data for personalized recommendations
ncf_model = keras.models.load_model('./app/models/ncf_model.h5')

with open('./app/models/user_id_to_idx.pkl', 'rb') as f:
    user_id_to_idx = pickle.load(f)

with open('./app/models/item_ids.pkl', 'rb') as f:
    item_ids = pickle.load(f)

with open('./app/models/item_id_to_title.pkl', 'rb') as f:
    item_id_to_title = pickle.load(f)

with open('./app/models/user_ids.pkl', 'rb') as f:
    user_ids = pickle.load(f)

data = pd.read_pickle('./app/models/data.pkl')

# Reverse mapping from item_idx to item_id
item_idx_to_id = {idx: item_id for idx, item_id in enumerate(item_ids)}

# Create item_id_to_idx mapping
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

# Get number of users and items
num_users = len(user_ids)
num_items = len(item_ids)

# Home route
@app.route('/')
def home():
    return render_template('home.html', movies=movie_titles, user_ids=user_ids[:100])

# Route for SVD-based movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form['movie_name']
    num_recommendations = int(request.form.get('num_recommendations', 5))

    if selected_movie not in movie_indices:
        error_message = "Movie not found in the database."
        return render_template('recommendations.html', error_message=error_message)

    # Get recommendations
    recommendations = get_similar_movies_svd(selected_movie, num_recommendations)
    return render_template('recommendations.html', movie_name=selected_movie, recommendations=recommendations)

# Route for similar movies
@app.route('/similar_movies/<movie_name>')
def similar_movies(movie_name):
    num_recommendations = 5  # Number of similar movies to show

    # Check if the movie exists in the indices
    if movie_name not in movie_indices:
        error_message = "Movie not found in the database."
        return render_template('recommendations.html', error_message=error_message)

    # Get similar movies
    recommendations = get_similar_movies_svd(movie_name, num_recommendations)

    return render_template('similar_movies.html', movie_name=movie_name, recommendations=recommendations)

# Route for NCF-based personalized recommendations
@app.route('/personalized_recommend', methods=['POST'])
def personalized_recommend():
    selected_user_id = int(request.form['user_id'])
    num_recommendations = int(request.form.get('num_recommendations', 10))

    liked_movies = get_liked_movies(selected_user_id, num_movies=5)
    disliked_movies = get_disliked_movies(selected_user_id, num_movies=5)
    recommendations = recommend_movies(selected_user_id, num_recommendations)

    return render_template(
        'personalized_recommendations.html',
        user_id=selected_user_id,
        liked_movies=liked_movies,
        disliked_movies=disliked_movies,
        recommendations=recommendations
    )

# Route to start rating movies
@app.route('/rate_movies', methods=['GET', 'POST'])
def rate_movies():
    if 'rated_movies' not in session:
        session['rated_movies'] = {}
        session['seen_movies'] = []

    error_message = None  # Initialize error_message

    if request.method == 'POST':
        # Retrieve ratings from form
        for key, value in request.form.items():
            if key.startswith('movie_'):
                movie_title = key.replace('movie_', '')
                if value:  # Check if a rating was provided
                    rating = int(value)
                    rated_movies = session.get('rated_movies', {})
                    rated_movies[movie_title] = rating
                    session['rated_movies'] = rated_movies

                    seen_movies = session.get('seen_movies', [])
                    if movie_title not in seen_movies:
                        seen_movies.append(movie_title)
                        session['seen_movies'] = seen_movies

        # Determine which button was clicked
        action = request.form.get('action')
        if action == 'done':
            if len(session.get('rated_movies', {})) >= 5:
                # Generate recommendations
                recommendations = generate_recommendations_from_ratings(session['rated_movies'])
                return render_template('session_recommendations.html', recommendations=recommendations)
            else:
                error_message = "Please rate at least 5 movies before proceeding."
                # Instead of changing pages, render the same page with an error message

        # Continue to next set of movies or show error
        # Select movies to rate
        rated_movie_titles = session.get('seen_movies', [])
        movies_to_rate = select_movies_for_rating(rated_movie_titles)

        # If no more movies to rate
        if not movies_to_rate:
            if len(session.get('rated_movies', {})) >= 5:
                # Generate recommendations
                recommendations = generate_recommendations_from_ratings(session['rated_movies'])
                return render_template('session_recommendations.html', recommendations=recommendations)
            else:
                error_message = "No more movies to rate. Please rate at least 5 movies."
                # Render the same page with an error message
                return render_template('rate_movies.html', movies=[], error_message=error_message)

        return render_template('rate_movies.html', movies=movies_to_rate, error_message=error_message)

    else:
        # GET request, display movies to rate
        rated_movie_titles = session.get('seen_movies', [])
        movies_to_rate = select_movies_for_rating(rated_movie_titles)
        return render_template('rate_movies.html', movies=movies_to_rate, error_message=error_message)

# Route to reset ratings
@app.route('/reset_ratings')
def reset_ratings():
    session.pop('rated_movies', None)
    session.pop('seen_movies', None)
    return redirect(url_for('rate_movies'))

# SVD-based recommendation function
def get_similar_movies_svd(movie_name, num_movies=5):
    # Find the index of the movie
    movie_idx = movie_indices[movie_name]

    # Get the vector for the specified movie in the reduced feature space
    movie_vec = svd_matrix[movie_idx].reshape(1, -1)

    # Compute cosine similarity with other movies in the reduced space
    cosine_sim = cosine_similarity(movie_vec, svd_matrix)[0]

    # Get indices of the top similar movies
    similar_indices = np.argsort(cosine_sim)[-num_movies-1:-1][::-1]

    # Get the titles of the most similar movies
    similar_movies = [movie_titles[i] for i in similar_indices]

    return similar_movies

# Function to get movies a user liked
def get_liked_movies(user_id, num_movies=10):
    user_data = data[(data['user_id'] == user_id) & (data['rating'] >= 4)]
    liked_movies = user_data['title'].tolist()
    if num_movies > len(liked_movies):
        num_movies = len(liked_movies)
    liked_movies = random.sample(liked_movies, num_movies)
    return liked_movies

# Function to get movies a user disliked
def get_disliked_movies(user_id, num_movies=10):
    user_data = data[(data['user_id'] == user_id) & (data['rating'] <= 2)]
    disliked_movies = user_data['title'].tolist()
    if num_movies > len(disliked_movies):
        num_movies = len(disliked_movies)
    disliked_movies = random.sample(disliked_movies, num_movies)
    return disliked_movies

# NCF-based recommendation function for existing users
def recommend_movies(user_id, num_recommendations=10):
    user_idx = user_id_to_idx.get(user_id)
    if user_idx is None or user_idx >= num_users:
        print("User ID not found or out of range.")
        return []

    # Items the user has interacted with
    user_data = data[data['user_idx'] == user_idx]
    interacted_items = set(user_data['item_idx'].tolist())

    # Filter items that are out of the range for item indices
    all_items = set(range(num_items))
    items_to_predict = list(all_items - interacted_items)

    # Predict interaction scores
    user_array = np.full(len(items_to_predict), user_idx)
    item_array = np.array(items_to_predict)

    # Predict
    predictions = ncf_model.predict([user_array, item_array], batch_size=128).flatten()

    # Get top N items
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    recommended_item_idxs = [items_to_predict[i] for i in top_indices]

    # Map item indices to titles
    recommended_item_ids = [item_idx_to_id[idx] for idx in recommended_item_idxs]
    recommended_titles = [item_id_to_title.get(item_id, 'Unknown') for item_id in recommended_item_ids]

    return recommended_titles

# Function to select movies for rating, influenced by previous ratings
def select_movies_for_rating(rated_movie_titles=None):
    if rated_movie_titles is None:
        rated_movie_titles = []
    # Exclude movies already rated
    available_movie_indices = [idx for idx in range(len(movie_titles)) if movie_titles[idx] not in rated_movie_titles]

    # If less than required movies are left, return them all
    if len(available_movie_indices) <= 5:
        selected_indices = available_movie_indices
    else:
        # If user has rated movies, select new movies based on similarities
        if session.get('rated_movies'):
            rated_movies = list(session['rated_movies'].keys())
            # Get indices of rated movies
            rated_movie_indices = [movie_indices[movie] for movie in rated_movies if movie in movie_indices]
            # Get user ratings
            ratings = np.array([session['rated_movies'][movie] for movie in rated_movies])
            # Compute weighted average vector of rated movies
            rated_movie_vectors = svd_matrix[rated_movie_indices]
            user_profile = np.dot(ratings, rated_movie_vectors) / np.sum(ratings)
            # Compute similarities to all movies
            cosine_sim = cosine_similarity(user_profile.reshape(1, -1), svd_matrix)[0]
            # Exclude movies already seen
            unseen_movie_indices = [idx for idx in available_movie_indices]
            # Get similarities for unseen movies
            unseen_similarities = [(idx, cosine_sim[idx]) for idx in unseen_movie_indices]
            # Sort by similarity
            unseen_similarities.sort(key=lambda x: x[1], reverse=True)
            # Now, select movies based on bins
            selected_indices = []
            bins = [
                (0, 50, 2),     # Top 50 most similar, pick 2 movies
                (50, 100, 1),   # Ranks 51-100, pick 1 movie
                (100, 200, 1),  # Ranks 101-200, pick 1 movie
                (200, 500, 1)   # Ranks 201-500, pick 1 movie
            ]
            for start, end, num in bins:
                # Ensure the indices are within the range
                bin_movies = unseen_similarities[start:end]
                if len(bin_movies) >= num:
                    selected_bin_indices = random.sample(bin_movies, num)
                else:
                    selected_bin_indices = bin_movies  # Take whatever is available
                selected_indices.extend([idx for idx, sim in selected_bin_indices])
            # Shuffle the selected movies
            random.shuffle(selected_indices)
        else:
            # If no ratings yet, select random movies
            selected_indices = random.sample(available_movie_indices, 5)

    # Get movie titles
    selected_movies = [movie_titles[idx] for idx in selected_indices]

    return selected_movies

# Function to generate recommendations from user ratings
def generate_recommendations_from_ratings(user_ratings, num_recommendations=10):
    # Prepare item indices and ratings
    item_idxs = []
    ratings_list = []
    for movie_title, rating in user_ratings.items():
        if movie_title in movie_indices:
            movie_idx = movie_indices[movie_title]
            item_idxs.append(movie_idx)
            ratings_list.append(rating)

    if not item_idxs:
        return []

    # Generate user embedding based on rated movies
    item_embedding_layer = ncf_model.get_layer('item_embedding')
    item_embeddings = item_embedding_layer.get_weights()[0]
    user_vector = np.average(item_embeddings[item_idxs], axis=0, weights=ratings_list)

    # Compute scores for all items
    scores = np.dot(item_embeddings, user_vector)

    # Exclude items already rated
    scores[item_idxs] = -np.inf

    # Get top N items
    top_indices = np.argsort(scores)[-num_recommendations:][::-1]
    recommended_titles = [movie_titles[idx] for idx in top_indices]

    return recommended_titles

if __name__ == '__main__':
    app.run(debug=True)
