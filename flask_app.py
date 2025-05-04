from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import logging
from pathlib import Path
from fuzzywuzzy import process
import requests
import zipfile
import io
import json
import time
import atexit
from urllib.parse import quote
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Check if TMDB_API_KEY is set
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    print("Error: TMDB_API_KEY is not set. Please check your .env file or environment variables.")
    # sys.exit(1)

print("TMDB_API_KEY is loaded successfully.")


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Global variables
model = None
movies_df = None
tfidf_matrix = None
movie_indices = None
poster_cache = {}

def download_dataset():
    """Download MovieLens dataset if not available locally."""
    data_path = Path('data')
    data_path.mkdir(exist_ok=True)
    
    movies_path = data_path / 'movies.csv'
    
    if not movies_path.exists():
        logger.info("Downloading MovieLens dataset...")
        try:
            # MovieLens small dataset URL (100k)
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            
            # Download the zip file
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Extract the zip file
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(data_path)
                
                # Move files to the right location
                extracted_folder = data_path / 'ml-latest-small'
                if (extracted_folder / 'movies.csv').exists():
                    os.rename(extracted_folder / 'movies.csv', movies_path)
                    logger.info("Dataset downloaded and extracted successfully")
                    
                    # Clean up extracted folder
                    if os.path.exists(extracted_folder):
                        import shutil
                        shutil.rmtree(extracted_folder)
                else:
                    logger.error("Expected files not found in the downloaded archive")
                    return False
            else:
                logger.error(f"Failed to download dataset: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False
    
    return True

def preprocess_data():
    """Preprocess the movies data to prepare for recommendation."""
    global movies_df
    
    if movies_df is None:
        logger.error("No data to preprocess")
        return False
    
    try:
        # Extract year from title and create clean title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').fillna('')
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        # Process genres - replace pipes with spaces for TF-IDF
        if 'genres' in movies_df.columns:
            movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
        else:
            logger.warning("No 'genres' column found in dataset. Using empty values.")
            movies_df['genres'] = ''
        
        # Fill missing values
        movies_df = movies_df.fillna('')
        
        logger.info("Data preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False

def load_data():
    """Load movie data from CSV file."""
    global movies_df
    
    data_path = Path('data')
    data_path.mkdir(exist_ok=True)
    
    movies_path = data_path / 'movies.csv'
    
    # Try to download the dataset if it doesn't exist
    if not movies_path.exists() and not download_dataset():
        logger.warning("Dataset download failed. Creating sample data for demonstration.")
        # Create sample data if the real dataset is missing
        sample_data = {
            'movieId': list(range(1, 11)),
            'title': [
                'The Shawshank Redemption (1994)',
                'The Godfather (1972)',
                'The Dark Knight (2008)',
                'Pulp Fiction (1994)',
                'Fight Club (1999)',
                'Forrest Gump (1994)',
                'Inception (2010)',
                'The Matrix (1999)',
                'Goodfellas (1990)',
                'The Silence of the Lambs (1991)'
            ],
            'genres': [
                'Drama',
                'Crime Drama',
                'Action Crime Drama',
                'Crime Drama Thriller',
                'Drama Thriller',
                'Drama Romance',
                'Action Sci-Fi Thriller',
                'Action Sci-Fi',
                'Crime Drama',
                'Crime Drama Thriller'
            ]
        }
        movies_df = pd.DataFrame(sample_data)
        # Save sample data
        movies_df.to_csv(movies_path, index=False)
        logger.info("Created sample data with 10 movies")
    else:
        try:
            # Load movies dataset
            movies_df = pd.read_csv(movies_path)
            logger.info(f"Loaded {len(movies_df)} movies from existing dataset")
            
            # Additional steps to load ratings data if available
            ratings_path = data_path / 'ratings.csv'
            if ratings_path.exists():
                try:
                    ratings_df = pd.read_csv(ratings_path)
                    logger.info(f"Loaded {len(ratings_df)} ratings")
                    
                    # Calculate average rating for each movie
                    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
                    avg_ratings.columns = ['movieId', 'avg_rating']
                    
                    # Merge ratings with movies
                    movies_df = pd.merge(movies_df, avg_ratings, on='movieId', how='left')
                    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0).round(1)
                    logger.info("Added average ratings to movies")
                except Exception as e:
                    logger.warning(f"Error loading ratings data: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading movies data: {str(e)}")
            return None
    
    # Preprocess the loaded data
    if not preprocess_data():
        logger.warning("Data preprocessing failed. Using data as-is.")
    
    return movies_df

def enrich_movie_data():
    """Add additional data to the movies dataframe."""
    global movies_df
    
    if movies_df is None:
        logger.error("No movie data to enrich")
        return
    
    try:
        # Create a popularity metric (can be based on number of ratings if available)
        data_path = Path('data')
        ratings_path = data_path / 'ratings.csv'
        
        if ratings_path.exists():
            ratings_df = pd.read_csv(ratings_path)
            # Count number of ratings per movie
            rating_counts = ratings_df['movieId'].value_counts().reset_index()
            rating_counts.columns = ['movieId', 'rating_count']
            
            # Add to movies_df
            movies_df = pd.merge(movies_df, rating_counts, on='movieId', how='left')
            movies_df['rating_count'] = movies_df['rating_count'].fillna(0).astype(int)
            
            # Calculate popularity score
            max_count = movies_df['rating_count'].max()
            if max_count > 0:  # Avoid division by zero
                movies_df['popularity'] = (movies_df['rating_count'] / max_count * 10).round(1)
            else:
                movies_df['popularity'] = 0
                
            logger.info("Added popularity scores based on rating counts")
        else:
            # If no ratings, assign random popularity for demo purposes
            movies_df['popularity'] = np.random.uniform(1, 10, size=len(movies_df)).round(1)
            logger.info("Added random popularity scores (no ratings data found)")
            
        # Extract decades for decade-based filtering
        movies_df['decade'] = movies_df['year'].apply(
            lambda x: f"{x[:3]}0s" if x.isdigit() and len(x) == 4 else "Unknown"
        )
        
        logger.info("Movie data enrichment completed")
        
    except Exception as e:
        logger.error(f"Error enriching movie data: {str(e)}")

def train_model():
    """Train the TF-IDF model for recommendations."""
    global tfidf_matrix, movie_indices, movies_df, model
    
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    
    tfidf_path = model_path / 'tfidf_model.pkl'
    matrix_path = model_path / 'tfidf_matrix.pkl'
    
    if tfidf_path.exists() and matrix_path.exists() and movies_df is not None:
        logger.info("Loading pre-trained TF-IDF model...")
        try:
            with open(tfidf_path, 'rb') as f:
                tfidf = pickle.load(f)
            with open(matrix_path, 'rb') as f:
                tfidf_matrix = pickle.load(f)
                
            # Verify the matrix dimensions match the current dataset
            if tfidf_matrix.shape[0] != len(movies_df):
                logger.warning("TF-IDF matrix dimensions don't match current dataset. Retraining model.")
                raise ValueError("Matrix dimensions mismatch")
        except Exception as e:
            logger.warning(f"Error loading model: {str(e)}. Training new model.")
            tfidf = TfidfVectorizer(stop_words='english')
            # Create content field for better recommendations
            movies_df['content'] = movies_df['genres'] + ' ' + movies_df['year']
            tfidf_matrix = tfidf.fit_transform(movies_df['content'])
            with open(tfidf_path, 'wb') as f:
                pickle.dump(tfidf, f)
            with open(matrix_path, 'wb') as f:
                pickle.dump(tfidf_matrix, f)
    else:
        logger.info("Training new TF-IDF model...")
        tfidf = TfidfVectorizer(stop_words='english')
        # Create content field for better recommendations
        movies_df['content'] = movies_df['genres'] + ' ' + movies_df['year']
        tfidf_matrix = tfidf.fit_transform(movies_df['content'])
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf, f)
        with open(matrix_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
    
    movie_indices = pd.Series(movies_df.index, index=movies_df['title_clean'])
    model = {'tfidf': tfidf, 'matrix': tfidf_matrix}
    logger.info("Model ready")
    return model

def fuzzy_match_movie(title, threshold=0.7):
    """Find movie titles that match the query with fuzzy matching."""
    global movies_df, movie_indices
    
    # Check if we have data
    if movies_df is None or len(movies_df) == 0:
        return []
    
    # Get list of movie titles
    movie_titles = movies_df['title_clean'].tolist()
    
    # Find matches
    matches = process.extractBests(title, movie_titles, score_cutoff=threshold * 100, limit=5)
    
    if not matches:
        return []
    
    # Return matched movies
    matched_movies = []
    for match, score in matches:
        idx = movie_indices[match]
        movie = movies_df.iloc[idx]
        matched_movies.append({
            'id': int(movie.name),
            'title': movie['title'],
            'genres': movie['genres'].replace(' ', ', '),
            'year': movie['year'],
            'score': score/100
        })
    
    return matched_movies

def get_recommendations(title, n=10, genre_filter=None):
    """Generate n movie recommendations based on movie title with optional genre filtering."""
    global tfidf_matrix, movie_indices, movies_df
    
    # Check if we have data and model
    if movies_df is None or tfidf_matrix is None:
        return []
    
    # First, find the best match for the title
    matches = fuzzy_match_movie(title)
    
    if not matches:
        return []
    
    # Use the best match
    best_match = matches[0]
    idx = best_match['id']
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n most similar movies (skip the first as it's the input movie)
    sim_scores = sim_scores[1:n+50]  # Get more to allow for filtering
    
    # Filter by genre if specified
    recommendations = []
    movie_indices_list = [i[0] for i in sim_scores]
    
    for i, idx in enumerate(movie_indices_list):
        movie = movies_df.iloc[idx]
        
        # Apply genre filter if specified
        if genre_filter and genre_filter.lower() not in movie['genres'].lower():
            continue
            
        # Add to recommendations
        movie_data = {
            'id': int(movie.name),
            'title': movie['title'],
            'genres': movie['genres'].replace(' ', ', '),
            'year': movie['year'],
            'similarity': sim_scores[i][1]
        }
        
        # Add rating if available
        if 'avg_rating' in movie:
            movie_data['rating'] = float(movie['avg_rating'])
            
        # Add popularity if available
        if 'popularity' in movie:
            movie_data['popularity'] = float(movie['popularity'])
            
        recommendations.append(movie_data)
        
        # Stop once we have enough recommendations
        if len(recommendations) >= n:
            break
    
    return recommendations

def get_movie_poster(movie_title, year=None):
    """Fetch movie poster URL from TMDB API."""
    # Get API key from environment variables
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
    
    if not TMDB_API_KEY:
        logger.warning("TMDB API key not found in environment variables")
        return "/static/img/movie-placeholder.svg"
    
    # Clean up title - remove year if present in title
    if year and year.isdigit():
        search_title = movie_title
    else:
        # Extract year from title if not provided separately
        year_match = re.search(r'\((\d{4})\)', movie_title)
        if year_match:
            year = year_match.group(1)
            search_title = re.sub(r'\s*\(\d{4}\)', '', movie_title)
        else:
            search_title = movie_title
    
    # First, search for the movie
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={quote(search_title)}"
    if year and year.isdigit():
        search_url += f"&year={year}"
    
    try:
        response = requests.get(search_url)
        data = response.json()
        
        # Check if we got results
        if data.get('results') and len(data['results']) > 0:
            # Get the first movie result
            movie = data['results'][0]
            poster_path = movie.get('poster_path')
            
            if poster_path:
                # Construct full poster URL
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return poster_url
    except Exception as e:
        logger.error(f"Error fetching movie poster for {movie_title}: {str(e)}")
    
    # Return placeholder if no poster found
    return "/static/img/movie-placeholder.svg"

def get_cached_poster(movie_title, year=None):
    """Get movie poster URL with caching."""
    cache_key = f"{movie_title}_{year}" if year else movie_title
    
    # Check if poster URL is in cache
    if cache_key in poster_cache:
        return poster_cache[cache_key]
    
    # Get poster URL
    poster_url = get_movie_poster(movie_title, year)
    
    # Cache the result
    poster_cache[cache_key] = poster_url
    
    return poster_url

def save_poster_cache():
    """Save poster cache to file."""
    try:
        cache_path = Path('data')
        cache_path.mkdir(exist_ok=True)
        
        with open(cache_path / 'poster_cache.json', 'w') as f:
            json.dump(poster_cache, f)
        logger.info(f"Saved {len(poster_cache)} poster URLs to cache")
    except Exception as e:
        logger.error(f"Error saving poster cache: {str(e)}")

def load_poster_cache():
    """Load poster cache from file."""
    global poster_cache
    try:
        cache_path = Path('data/poster_cache.json')
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                poster_cache = json.load(f)
                logger.info(f"Loaded {len(poster_cache)} cached poster URLs")
    except Exception as e:
        logger.error(f"Error loading poster cache: {str(e)}")
        poster_cache = {}

# API Routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Serve the Vue frontend."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by title."""
    query = request.args.get('query', '')
    if not query or len(query) < 2:
        return jsonify({'results': []})
    
    results = fuzzy_match_movie(query)
    
    # Add poster URLs to results
    for movie in results:
        # movie['poster'] = get_cached_poster(movie['title'], movie['year'])
        movie['poster'] = None
    
    return jsonify({'results': results})

@app.route('/api/recommendations', methods=['GET'])
def recommend():
    """Get movie recommendations."""
    movie_title = request.args.get('title', '')
    count = request.args.get('count', 10, type=int)
    genre = request.args.get('genre', None)
    
    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400
    
    recommendations = get_recommendations(movie_title, count, genre)
    
    # Add poster URLs to recommendations
    for movie in recommendations:
        # movie['poster'] = get_cached_poster(movie['title'], movie['year'])
        movie['poster'] = None
    
    return jsonify({'query': movie_title, 'recommendations': recommendations})


@app.route('/api/get_poster', methods=['GET'])
def fetch_poster():
    """Fetch movie poster on demand."""
    title = request.args.get('title', '')
    year = request.args.get('year', None)

    if not title:
        return jsonify({'error': 'No movie title provided'}), 400

    poster_url = get_cached_poster(title, year)
    return jsonify({'poster': poster_url})



@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Return trending movies based on popularity."""
    global movies_df
    
    if movies_df is None or len(movies_df) == 0:
        return jsonify({'error': 'Movie data not loaded'}), 500
    
    # If we have popularity data, use it
    if 'popularity' in movies_df.columns:
        trending = movies_df.sort_values('popularity', ascending=False).head(10)
    else:
        # Otherwise, return random movies
        sample_size = min(10, len(movies_df))
        trending = movies_df.sample(sample_size)
    
    results = []
    for _, movie in trending.iterrows():
        movie_data = {
            'id': int(movie.name),
            'title': movie['title'],
            'genres': movie['genres'].replace(' ', ', '),
            'year': movie['year']
        }
        
        # Add rating if available
        if 'avg_rating' in movie:
            movie_data['rating'] = float(movie['avg_rating'])
            
        # Add popularity if available
        if 'popularity' in movie:
            movie_data['popularity'] = float(movie['popularity'])
        
        # Add poster URL
        movie_data['poster'] = get_cached_poster(movie_data['title'], movie_data['year'])
            
        results.append(movie_data)
        
    return jsonify({'trending': results})

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Return list of all unique genres."""
    global movies_df
    
    if movies_df is None or len(movies_df) == 0:
        return jsonify({'error': 'Movie data not loaded'}), 500
    
    # Extract all genres from the dataset
    all_genres = []
    for genres in movies_df['genres'].str.split(' '):
        all_genres.extend(genres)
    
    # Get unique genres and sort
    unique_genres = sorted(list(set([g for g in all_genres if g])))
    
    return jsonify({'genres': unique_genres})

@app.route('/api/decades', methods=['GET'])
def get_decades():
    """Return list of all movie decades."""
    global movies_df
    
    if movies_df is None or len(movies_df) == 0 or 'decade' not in movies_df.columns:
        return jsonify({'error': 'Movie data not available'}), 500
    
    # Get unique decades and sort
    decades = sorted(list(movies_df['decade'].unique()))
    # Remove 'Unknown' from the list
    if 'Unknown' in decades:
        decades.remove('Unknown')
    
    return jsonify({'decades': decades})

def initialize():
    """Initialize app by loading data, enriching it and training model."""
    global movies_df, model
    try:
        movies_df = load_data()
        if movies_df is not None:
            enrich_movie_data()
            model = train_model()
            load_poster_cache()
            logger.info("Application initialized successfully")
        else:
            logger.error("Failed to load movie data")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

@app.route('/api/initialize', methods=['GET'])
def initialize_app():
    """Endpoint to initialize the app if it wasn't initialized on startup."""
    initialize()
    return jsonify({'status': 'success', 'message': 'App initialized successfully'})

# Register shutdown handler to save cache
atexit.register(save_poster_cache)

# Initialize on startup
with app.app_context():
    initialize()

if __name__ == '__main__':
    app.run(debug=True)