from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate secret key programmatically

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

    if request.method == 'POST':
        # Retrieve ratings from form
        for key, value in request.form.items():
            if key.startswith('movie_'):
                movie_title = key.replace('movie_', '')
                if value:  # Check if a rating was provided
                    rating = int(value)
                    session['rated_movies'][movie_title] = rating
                if movie_title not in session['seen_movies']:
                    session['seen_movies'].append(movie_title)

        # Check if 'done' button was clicked
        if 'done' in request.form:
            if len(session['rated_movies']) >= 5:
                # Generate recommendations
                recommendations = generate_recommendations_from_ratings(session['rated_movies'])
                return render_template('session_recommendations.html', recommendations=recommendations)
            else:
                error_message = "Please rate at least 5 movies before proceeding."
                return render_template('rate_movies.html', movies=[], error_message=error_message)

    # Select movies to rate
    rated_movie_titles = session['seen_movies']
    movies_to_rate = select_movies_for_rating(rated_movie_titles)

    # If no more movies to rate
    if not movies_to_rate:
        if len(session['rated_movies']) >= 5:
            # Generate recommendations
            recommendations = generate_recommendations_from_ratings(session['rated_movies'])
            return render_template('session_recommendations.html', recommendations=recommendations)
        else:
            error_message = "No more movies to rate. Please rate at least 5 movies."
            return render_template('rate_movies.html', movies=[], error_message=error_message)

    return render_template('rate_movies.html', movies=movies_to_rate, error_message=None)

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


# Function to get similar movie indices
def get_similar_movie_indices(movie_name):
    if movie_name not in movie_indices:
        return []
    movie_idx = movie_indices[movie_name]
    movie_vec = svd_matrix[movie_idx].reshape(1, -1)
    cosine_sim = cosine_similarity(movie_vec, svd_matrix)[0]
    similar_indices = np.argsort(cosine_sim)[::-1]
    return similar_indices

# Function to select movies for rating
def select_movies_for_rating(rated_movie_titles):
    # Exclude movies already rated
    available_movie_indices = [idx for idx in range(len(movie_titles)) if movie_titles[idx] not in rated_movie_titles]

    # If less than 5 movies are left, return them all
    if len(available_movie_indices) <= 5:
        selected_indices = available_movie_indices
    else:
        # Randomly select movies based on specified criteria
        selected_indices = []

        # First movie: from top 25 movies
        first_movie_idx = random.choice(available_movie_indices[:25])
        selected_indices.append(first_movie_idx)

        # Second movie: from 25-50
        second_movie_idx = random.choice(available_movie_indices[25:50])
        selected_indices.append(second_movie_idx)

        # Third movie: from 50-100
        third_movie_idx = random.choice(available_movie_indices[50:100])
        selected_indices.append(third_movie_idx)

        # Fourth movie: from 100-500
        if len(available_movie_indices) >= 500:
            fourth_movie_idx = random.choice(available_movie_indices[100:500])
        else:
            fourth_movie_idx = random.choice(available_movie_indices[100:])
        selected_indices.append(fourth_movie_idx)

        # Fifth movie: from the 100 most dissimilar movies
        fifth_movie_idx = random.choice(available_movie_indices[-100:])
        selected_indices.append(fifth_movie_idx)

        # Shuffle the selected movies
        random.shuffle(selected_indices)

    # Get movie titles
    selected_movies = [movie_titles[idx] for idx in selected_indices]

    return selected_movies

# Function to generate recommendations from user ratings
def generate_recommendations_from_ratings(user_ratings, num_recommendations=10):
    # Create a DataFrame from user_ratings
    user_ratings_df = pd.DataFrame({
        'user_id': [9999]*len(user_ratings),  # Use a unique user_id for the session
        'item_id': [item_ids[item_id_to_idx.get(movie_indices[movie])] for movie in user_ratings.keys() if movie in movie_indices],
        'rating': list(user_ratings.values()),
    })
    user_ratings_df['user_idx'] = len(user_id_to_idx)  # New index for the new user
    user_ratings_df['item_idx'] = [item_id_to_idx.get(movie_indices[movie]) for movie in user_ratings.keys() if movie in movie_indices]
    user_ratings_df['title'] = user_ratings_df['item_id'].map(item_id_to_title)

    # Filter out invalid indices for items and users
    user_ratings_df = user_ratings_df[(user_ratings_df['item_idx'] < num_items) & (user_ratings_df['user_idx'] < num_users)]

    # Combine with existing data
    combined_data = pd.concat([data, user_ratings_df], ignore_index=True)

    # Generate recommendations
    user_idx = len(user_id_to_idx)  # Index for the new user

    # Items the user has interacted with
    interacted_items = set(user_ratings_df['item_idx'].tolist())

    # Items not yet interacted with
    all_items = set(range(num_items))
    items_to_predict = list(all_items - interacted_items)

    # Predict interaction scores, ensuring valid indices
    user_array = np.full(len(items_to_predict), user_idx)
    item_array = np.array(items_to_predict)

    # Filter valid indices
    user_array = np.array([idx for idx in user_array if idx < num_users])
    item_array = np.array([idx for idx in item_array if idx < num_items])

    predictions = ncf_model.predict([user_array, item_array], batch_size=32).flatten()

    # Get top N items
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    recommended_item_idxs = [items_to_predict[i] for i in top_indices]

    # Map item indices to titles
    recommended_item_ids = [item_idx_to_id[idx] for idx in recommended_item_idxs]
    recommended_titles = [item_id_to_title.get(item_id, 'Unknown') for item_id in recommended_item_ids]

    return recommended_titles


if __name__ == '__main__':
    app.run(debug=True)
