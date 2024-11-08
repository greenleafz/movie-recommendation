from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved SVD matrix
with open('./models/svd_matrix.pkl', 'rb') as f:
    svd_matrix = pickle.load(f)

# Load movie titles
with open('./models/movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

# Convert movie titles to a list if not already
if isinstance(movie_titles, np.ndarray):
    movie_titles = movie_titles.tolist()

# Create a mapping from movie titles to indices
movie_indices = {title: idx for idx, title in enumerate(movie_titles)}

@app.route('/')
def home():
    return render_template('home.html', movies=movie_titles)

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

if __name__ == '__main__':
    app.run(debug=True)
