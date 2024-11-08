# AI-Powered Movie Recommendation System

This is a simple movie recommendation system built using **Singular Value Decomposition (SVD)**. The application provides movie recommendations based on user ratings from the MovieLens dataset and is deployed as a web app using Flask.

## Table of Contents
- [AI-Powered Movie Recommendation System](#ai-powered-movie-recommendation-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Setup Instructions](#setup-instructions)
  - [Usage](#usage)
  - [Deployment](#deployment)
  - [Screenshots](#screenshots)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Overview
This project uses SVD to reduce the dimensionality of the user-movie ratings matrix, capturing latent features to make movie recommendations. A randomly selected movie serves as the basis for suggesting similar movies. The app is built with Python and Flask and is deployed on Heroku.

## Features
- Provides movie recommendations based on user ratings.
- Allows users to select a movie and get a list of similar movies.
- Deployed as a web application with a simple user interface.

## Technologies Used
- **Python**: Programming language for data processing and building the web application.
- **Pandas**: Data manipulation and handling.
- **NumPy**: Numerical computations and handling matrices.
- **scikit-learn**: Machine learning library for performing SVD.
- **Flask**: Web framework for creating the application.
- **Heroku**: Cloud platform for deployment.

## Setup Instructions
To run this project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  ️ # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Data**:
   - Download the MovieLens 100K dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k/).
   - Place the downloaded data files in the `data/movielens/` directory.

5. **Run the Application**:
   ```bash
   python app/app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage
1. Open the app in your web browser.
2. Select a movie from the dropdown menu on the homepage.
3. Specify the number of similar movies you would like to see.
4. Click "Get Recommendations" to see movies similar to the one selected.

## Deployment
The app is deployed on **Heroku**. You can access the live version here: [Your Heroku App Link](https://your-app-name.herokuapp.com)

## Screenshots
To be added as the application is further developed.

## Acknowledgments
- **MovieLens Dataset**: Provided by GroupLens Research at the University of Minnesota. This dataset was used for training and evaluating the recommendation model.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
