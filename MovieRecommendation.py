import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

ratings_path = '/workspaces/MoviePrediction/ratings.csv'
movies_path = '/workspaces/MoviePrediction/movies.csv'

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path, usecols=[0, 1], encoding='latin-1')

user_sample = ratings['userId'].sample(n=1000, random_state=42)
ratings = ratings[ratings['userId'].isin(user_sample)]

data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_movies_by_genre(movies_path, genre):
    # Read movies with genres this time
    movies_with_genres = pd.read_csv(movies_path, encoding='latin-1')
    # Filter movies that contain the specified genre
    genre_movies = movies_with_genres[movies_with_genres['genres'].str.contains(genre, case=False, na=False)]
    return genre_movies['title'].tolist()

def get_imdb_top_movies(n=100):
    # URL for IMDb Top 250 movies
    url = "https://www.imdb.com/chart/top/"
    
    try:
        # Send request with headers to avoid blocking
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get movie titles (adjust selectors based on current IMDb HTML structure)
        movies_list = []
        movie_elements = soup.select('td.titleColumn')
        
        for i in range(min(n, len(movie_elements))):
            title = movie_elements[i].get_text().strip()
            # Clean up the title (remove year and ranking)
            title = ' '.join(title.split()[1:-1])
            movies_list.append(title)
            
        return movies_list
    except Exception as e:
        print(f"Error fetching IMDb movies: {e}")
        return []

def get_user_ratings(num_movies=5):
    user_ratings = {}
    
    print("\nEnter 5 of your favorite movies and rate them (1-5):")
    for i in range(num_movies):
        while True:
            movie_title = input(f"Enter movie {i+1}: ").strip()
            if movie_title in user_movie_matrix.columns:
                try:
                    rating = float(input(f"Rate {movie_title} (1-5): "))
                    if 1 <= rating <= 5:
                        user_ratings[movie_title] = rating
                        break
                    else:
                        print("Rating should be between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5.")
            else:
                print("Movie not found in database. Please try another movie.")
    
    return user_ratings

def add_user_ratings(user_ratings, user_movie_matrix):
    matrix = user_movie_matrix.copy()
    
    new_user_id = len(matrix.index)
    matrix.loc[new_user_id] = 0
    
    for movie, rating in user_ratings.items():
        if movie in matrix.columns:
            matrix.at[new_user_id, movie] = rating
    
    # Calculate new similarity matrix
    user_similarity = cosine_similarity(matrix.fillna(0))
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=matrix.index, 
        columns=matrix.index
    )
    
    return new_user_id, matrix, user_similarity_df

def get_recommendations(user_id, num_recommendations=5):
    # Get the movies the user has already rated
    rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index.tolist()
    
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
    
    # Get movies from similar users, excluding already rated movies
    recommended_movies = (data[data['userId'].isin(similar_users)]['title']
                        .value_counts()
                        .loc[lambda x: ~x.index.isin(rated_movies)]  # Exclude rated movies
                        .index[:num_recommendations])
    
    return recommended_movies

user_ratings = get_user_ratings()
new_user_id, user_movie_matrix, user_similarity_df = add_user_ratings(user_ratings, user_movie_matrix)

print(get_recommendations(new_user_id))

