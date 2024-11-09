from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

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

# Get number of users and items
num_users = len(user_ids)
num_items = len(item_ids)

# Home route
@app.route('/')
def home():
    # Display a limited number of user IDs for practicality
    display_user_ids = user_ids[:100]
    return render_template('home.html', movies=movie_titles, user_ids=display_user_ids)

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

# NCF-based recommendation function
def recommend_movies(user_id, num_recommendations=10):
    user_idx = user_id_to_idx.get(user_id)
    if user_idx is None:
        print("User ID not found.")
        return []
    
    # Items the user has interacted with
    user_data = data[data['user_idx'] == user_idx]
    interacted_items = set(user_data['item_idx'].tolist())
    
    # Items not yet interacted with
    all_items = set(range(num_items))
    items_to_predict = list(all_items - interacted_items)
    
    # Predict interaction scores
    user_array = np.full(len(items_to_predict), user_idx)
    item_array = np.array(items_to_predict)
    
    predictions = ncf_model.predict([user_array, item_array], batch_size=1024).flatten()
    
    # Get top N items
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    recommended_item_idxs = [items_to_predict[i] for i in top_indices]
    
    # Map item indices to titles
    recommended_item_ids = [item_idx_to_id[idx] for idx in recommended_item_idxs]
    recommended_titles = [item_id_to_title[item_id] for item_id in recommended_item_ids]
    
    return recommended_titles



if __name__ == '__main__':
    app.run(debug=True)
