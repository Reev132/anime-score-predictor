# scripts/collection_config.py
"""
Configuration file for anime data collection
Easily adjust collection parameters here
"""

# Collection targets
COLLECTION_TARGETS = {
    'total_target': 1500,  # Target total anime count
    'top_anime_pages': 8,  # Pages of top anime (~200 anime)
    'seasonal_anime_limit': 200,  # Max seasonal anime
    'anime_per_season': 20,  # Anime per season
    'random_anime_count': 150,  # Random anime for diversity
    'anime_per_genre': 40,  # Anime per genre
    'anime_per_score_range': 100,  # Anime per score range
}

# Score ranges for balanced dataset
SCORE_RANGES = [
    (8.5, 10.0),   # Excellent (top-tier)
    (7.5, 8.4),    # Very good
    (6.5, 7.4),    # Good/Average
    (5.5, 6.4),    # Below average
    (0.0, 5.4),    # Poor (for learning what makes bad anime)
]

# Years to collect seasonal data from
COLLECTION_YEARS = list(range(2010, 2024))  # 2010-2023

# Seasons to collect
SEASONS = ['winter', 'spring', 'summer', 'fall']

# Genres to focus on (MAL genre IDs)
TARGET_GENRES = {
    1: "Action",
    2: "Adventure", 
    4: "Comedy",
    8: "Drama",
    9: "Ecchi",
    10: "Fantasy",
    14: "Horror",
    18: "Mecha",
    19: "Music",
    22: "Romance",
    23: "School",
    24: "Sci-Fi",
    25: "Shoujo",
    27: "Shounen",
    36: "Slice of Life",
    37: "Supernatural",
    41: "Suspense",
    42: "Award Winning",
    46: "Mystery",
}

# Anime types to include
INCLUDED_TYPES = ['TV', 'Movie', 'OVA', 'ONA', 'Special']

# Rate limiting (requests per minute)
RATE_LIMIT = {
    'requests_per_minute': 60,
    'max_retries': 3,
    'base_delay': 1.0,  # Base delay between requests
    'backoff_multiplier': 2,  # For exponential backoff
}

# Data quality filters
QUALITY_FILTERS = {
    'min_scored_by': 100,  # Minimum number of users who scored the anime
    'require_score': True,  # Only collect anime with scores
    'require_synopsis': False,  # Require synopsis (can be strict)
    'max_synopsis_length': 1000,  # Max synopsis length to save
    'max_background_length': 500,  # Max background length
}

# Additional data to collect
EXTRA_DATA = {
    'collect_staff': False,  # Collect director/staff info (slower)
    'collect_characters': False,  # Collect character info (much slower)
    'collect_recommendations': False,  # Collect recommendations (slower)
    'collect_reviews': False,  # Collect reviews (much slower)
}

# Output settings
OUTPUT_SETTINGS = {
    'include_synopsis': True,
    'include_background': True,
    'include_themes': True,
    'include_demographics': True,
    'sort_by_score': True,
    'remove_duplicates': True,
}