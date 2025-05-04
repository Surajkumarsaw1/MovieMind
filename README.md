# Movie Recommendation System

This project is a simple movie recommendation system built using Flask. It provides recommendations based on movie content, using TF-IDF.

## Features

*   Movie recommendations based on content similarity (TF-IDF on genres and year).
*   Web interface to search for movies and view recommendations.
*   Fuzzy title matching for robust search.
*   Fetches movie posters from The Movie Database (TMDB) API (Note: Currently commented out in `flask_app.py`).
*   Displays trending movies (based on rating counts if available).
*   API endpoints for searching, recommendations, genres, decades, and posters.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd movie-rec
    ```
    *(Replace `<repository_url>` with the actual URL of the repository)*

2.  **Create environment file (Optional - for Poster Fetching):**
    If you want to enable movie poster fetching (currently commented out in the code), create a file named `.env` in the project root directory and add your TMDB API key:
    ```
    TMDB_API_KEY=your_tmdb_api_key_here
    ```
    *Note: The application will run without this, but poster images will not be loaded.*

3.  **Install dependencies:**
    Install the required Python packages. A `requirements.txt` file is recommended. Key dependencies include:
    ```bash
    pip install Flask pandas numpy scikit-learn fuzzywuzzy python-Levenshtein requests python-dotenv
    ```
    *(Note: `python-Levenshtein` is recommended by `fuzzywuzzy` for better performance)*

4.  **Data and Model Setup (Automatic):**
    *   The application will automatically attempt to download the MovieLens Small dataset (`movies.csv`, `ratings.csv`) into the `data/` directory if it's not found on the first run.
    *   The TF-IDF model (`tfidf_model.pkl`, `tfidf_matrix.pkl`) will be trained and saved in the `models/` directory automatically if not present.

## Usage

1.  Run the Flask application:
    ```bash
    python flask_app.py
    ```

2.  Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address shown in the terminal).

3.  Use the web interface to search for movies and get recommendations.

## Project Structure

```
.
├── .env                 # Optional: For TMDB API Key
├── flask_app.py         # Main Flask application
├── data/
│   ├── movies.csv       # MovieLens movie data (auto-downloaded)
│   ├── ratings.csv      # MovieLens ratings data (auto-downloaded)
│   └── poster_cache.json # Cache for TMDB posters (if enabled)
├── models/
│   ├── tfidf_model.pkl  # Saved TF-IDF model
│   └── tfidf_matrix.pkl # Saved TF-IDF matrix
├── static/              # Static assets (CSS, images)
│   ├── css/
│   └── img/
│       └── movie-placeholder.svg
├── templates/
│   └── index.html       # Main HTML template
├── cache/               # (Potentially unused directory based on flask_app.py)
└── requirements.txt     # Recommended file for dependencies
```
*(Note: The `cache/` directory exists but doesn't seem to be actively used by `flask_app.py`. A `requirements.txt` file is recommended but not automatically generated.)*

## Data Files

*   **`data/movies.csv`**: Contains movie information (downloaded automatically if missing).
    *   Format: `movieId,title,genres`
    *   Example: `1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy`
*   **`data/ratings.csv`**: Contains user ratings for movies (downloaded automatically if missing). Used optionally to calculate average ratings and popularity.
    *   Format: `userId,movieId,rating,timestamp`
*   **`data/poster_cache.json`**: Caches movie poster URLs fetched from the TMDB API (if enabled) to reduce API calls. Automatically created and updated when poster fetching is active.
*   **`models/tfidf_model.pkl`**: Saved TF-IDF vectorizer model.
*   **`models/tfidf_matrix.pkl`**: Saved TF-IDF matrix representing movie content.

## Technologies Used

*   Python
*   Flask (Web Framework)
*   Pandas (Data Manipulation)
*   Scikit-learn (TF-IDF, Cosine Similarity)
*   Fuzzywuzzy (Fuzzy String Matching)
*   Requests (HTTP requests for TMDB API - *Optional, for poster fetching*)
*   Python-dotenv (Environment variable management - *Optional, for poster fetching*)
*   Pickle (Model persistence)
*   HTML/CSS/JavaScript (Frontend - served by Flask)

## License

*(Add license information here if applicable, e.g., MIT, Apache 2.0)*
