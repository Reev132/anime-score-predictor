import requests
import json
import time
import random
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import os
import re

class AnimeAPIClient:
    """
    Wrapper for Jikan MyAnimeList API
    Provides easy access to anime data with caching, error handling, and rate limiting
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the API client
        
        Args:
            cache_dir: Directory to store cached API responses
        """
        self.base_url = "https://api.jikan.moe/v4"
        self.cache_dir = cache_dir
        self.request_delay = 1.0  # 1 second delay between requests
        self.last_request_time = 0
        self.cache_duration = 86400  # 24 hours in seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"🎌 Jikan API Client initialized")
        print(f"📁 Cache directory: {cache_dir}")
    
    def _sanitize_cache_key(self, cache_key: str) -> str:
        """Sanitize cache key to be a valid filename"""
        # Remove invalid characters for filenames
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', cache_key)
        # Limit length
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        sanitized_key = self._sanitize_cache_key(cache_key)
        return os.path.join(self.cache_dir, f"{sanitized_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """
        Load data from cache if it exists and is not expired
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
            if datetime.now() - cache_time > timedelta(seconds=self.cache_duration):
                os.remove(cache_path)  # Delete expired cache
                return None
            
            return cache_data.get('data')
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """
        Save data to cache
        
        Args:
            cache_key: Unique identifier for the data
            data: Data to cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"⚠️  Error saving to cache: {e}")
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to respect API limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            sleep_time = self.request_delay - elapsed + random.uniform(0, 0.2)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Optional[Dict]:
        """
        Make a request to the Jikan API with error handling and rate limiting
        
        Args:
            endpoint: API endpoint (e.g., 'anime/16498')
            params: Query parameters
            use_cache: Whether to use cached responses
            
        Returns:
            Response data or None if request fails
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Create cache key from endpoint and params
        if params:
            params_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
            cache_key = f"{endpoint}_{params_str}"
        else:
            cache_key = endpoint
        
        # Try loading from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to cache
            if use_cache and 'data' in data:
                self._save_to_cache(cache_key, data['data'])
            
            return data.get('data')
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def search_anime_by_name(self, anime_name: str, exact_match: bool = False) -> Optional[Dict]:
        """
        Search for an anime by name and return detailed information
        
        Args:
            anime_name: Name of the anime to search for
            exact_match: If True, try to find exact title match
            
        Returns:
            Dictionary with detailed anime information or None if not found
        """
        print(f"🔍 Searching for anime: {anime_name}")
        
        # Search for the anime
        search_results = self._make_request(
            "anime",
            {"q": anime_name, "limit": 10},  # Get top 10 results
            use_cache=True
        )
        
        if not search_results or len(search_results) == 0:
            print(f"❌ No anime found with name: {anime_name}")
            return None
        
        # Try to find best match
        best_match = None
        anime_name_lower = anime_name.lower().strip()
        
        for anime in search_results:
            title = anime.get('title', '').lower()
            title_english = anime.get('title_english', '').lower() if anime.get('title_english') else ''
            
            # Check for exact match first
            if title == anime_name_lower or title_english == anime_name_lower:
                best_match = anime
                print(f"✅ Found exact match: {anime.get('title')}")
                break
            
            # Check if search term is in title
            if anime_name_lower in title or anime_name_lower in title_english:
                if best_match is None:
                    best_match = anime
        
        # If no good match, use first result
        if best_match is None:
            best_match = search_results[0]
            print(f"⚠️  No exact match, using: {best_match.get('title')}")
        
        # Get the MAL ID and fetch full details
        mal_id = best_match['mal_id']
        
        # Fetch full details
        return self.get_anime_by_id(mal_id)
    
    def get_anime_by_id(self, mal_id: int) -> Optional[Dict]:
        """
        Get detailed anime information by MAL ID
        
        Args:
            mal_id: MyAnimeList ID of the anime
            
        Returns:
            Dictionary with complete anime information
        """
        print(f"📺 Fetching anime details for MAL ID: {mal_id}")
        
        anime_data = self._make_request(f"anime/{mal_id}", use_cache=True)
        
        if anime_data is None:
            return None
        
        # Extract and structure the data for ML model
        processed_data = self._process_anime_data(anime_data)
        return processed_data
    
    def _process_anime_data(self, anime_data: Dict) -> Dict:
        """
        Process raw anime data from API into format suitable for ML model
        
        Args:
            anime_data: Raw data from Jikan API
            
        Returns:
            Processed and structured anime data
        """
        # Extract genres and studios with error handling
        genres = [g['name'] for g in anime_data.get('genres', [])] if anime_data.get('genres') else []
        studios = [s['name'] for s in anime_data.get('studios', [])] if anime_data.get('studios') else []
        themes = [t['name'] for t in anime_data.get('themes', [])] if anime_data.get('themes') else []
        demographics = [d['name'] for d in anime_data.get('demographics', [])] if anime_data.get('demographics') else []
        
        # Determine if anime has released
        status = anime_data.get('status', 'Unknown')
        is_released = status in ['Finished Airing', 'Currently Airing']
        
        processed = {
            'mal_id': anime_data.get('mal_id'),
            'title': anime_data.get('title', ''),
            'title_english': anime_data.get('title_english', ''),
            'title_japanese': anime_data.get('title_japanese', ''),
            'score': anime_data.get('score'),
            'scored_by': anime_data.get('scored_by'),
            'rank': anime_data.get('rank'),
            'popularity': anime_data.get('popularity'),
            'members': anime_data.get('members'),
            'favorites': anime_data.get('favorites'),
            'episodes': anime_data.get('episodes'),
            'duration': anime_data.get('duration', ''),
            'year': anime_data.get('year'),
            'season': anime_data.get('season', 'Unknown'),
            'type': anime_data.get('type', 'Unknown'),
            'status': status,
            'rating': anime_data.get('rating', 'Unknown'),
            'source': anime_data.get('source', 'Unknown'),
            'genres': genres,
            'themes': themes,
            'demographics': demographics,
            'studios': studios,
            'is_released': is_released,
            'synopsis': (anime_data.get('synopsis', '')[:500] if anime_data.get('synopsis') else ''),
            'background': (anime_data.get('background', '')[:200] if anime_data.get('background') else ''),
            'url': anime_data.get('url', ''),
            'images': anime_data.get('images', {}).get('jpg', {}).get('image_url', ''),
        }
        
        return processed
    
    def get_anime_suggestions(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Get autocomplete suggestions for anime search
        
        Args:
            query: Partial or full anime name
            limit: Maximum number of suggestions to return
            
        Returns:
            List of anime suggestions with basic info
        """
        print(f"💡 Getting suggestions for: {query}")
        
        search_results = self._make_request(
            "anime",
            {"q": query, "limit": limit},
            use_cache=True
        )
        
        if not search_results:
            return []
        
        suggestions = []
        for anime in search_results[:limit]:
            suggestions.append({
                'mal_id': anime.get('mal_id'),
                'title': anime.get('title', ''),
                'title_english': anime.get('title_english', ''),
                'year': anime.get('year'),
                'score': anime.get('score'),
                'type': anime.get('type', ''),
                'episodes': anime.get('episodes'),
                'image_url': anime.get('images', {}).get('jpg', {}).get('image_url', ''),
                'status': anime.get('status', ''),
            })
        
        print(f"✅ Found {len(suggestions)} suggestions")
        return suggestions
    
    def estimate_missing_features(self, anime_data: Dict) -> Dict:
        """
        Estimate missing features for unreleased anime using historical patterns
        
        This is useful for very new anime that don't have scores or member counts yet
        
        Args:
            anime_data: Anime data (may have missing values)
            
        Returns:
            Updated anime data with estimated values
        """
        print("📊 Estimating missing features...")
        
        # Default estimations based on anime characteristics
        if anime_data.get('members') is None or anime_data.get('members') == 0:
            # Estimate members based on type and recency
            if anime_data.get('type') == 'TV':
                anime_data['members'] = 50000
            elif anime_data.get('type') == 'Movie':
                anime_data['members'] = 30000
            else:
                anime_data['members'] = 10000
        
        if anime_data.get('scored_by') is None or anime_data.get('scored_by') == 0:
            # scored_by is typically 50-70% of members
            anime_data['scored_by'] = int(anime_data.get('members', 50000) * 0.6)
        
        if anime_data.get('popularity') is None or anime_data.get('popularity') == 0:
            # Estimate popularity (lower is better)
            # New anime with major studios: higher popularity (lower rank number)
            if any(s in anime_data.get('studios', []) for s in ['MAPPA', 'Madhouse', 'Kyoto Animation', 'WIT Studio']):
                anime_data['popularity'] = 200
            else:
                anime_data['popularity'] = 1000
        
        if anime_data.get('favorites') is None:
            # Favorites is typically 5-10% of members
            anime_data['favorites'] = int(anime_data.get('members', 50000) * 0.07)
        
        if anime_data.get('rank') is None:
            anime_data['rank'] = 9999  # Placeholder for unranked
        
        # Ensure episodes is set for unreleased anime
        if anime_data.get('episodes') is None:
            # Default to 12 for TV (common season length)
            if anime_data.get('type') == 'TV':
                anime_data['episodes'] = 12
            elif anime_data.get('type') == 'Movie':
                anime_data['episodes'] = 1
            else:
                anime_data['episodes'] = 0
        
        print("✅ Features estimated successfully")
        return anime_data
    
    def get_full_anime_info_for_prediction(self, anime_name: str) -> Optional[Dict]:
        """
        Get complete anime information ready for ML model prediction
        This is the main function to call for getting prediction-ready data
        
        Args:
            anime_name: Name of the anime to search for
            
        Returns:
            Complete anime data ready for ML prediction, or None if not found
        """
        print(f"🎬 Fetching complete anime info for: {anime_name}")
        
        # Search for the anime
        anime_info = self.search_anime_by_name(anime_name)
        
        if anime_info is None:
            print(f"❌ Could not find anime: {anime_name}")
            return None
        
        # Estimate any missing features
        anime_info = self.estimate_missing_features(anime_info)
        
        print(f"✅ Successfully retrieved anime info for: {anime_info['title']}")
        return anime_info
    
    def get_random_anime(self) -> Optional[Dict]:
        """
        Get a random anime from the database
        Useful for testing and exploration
        
        Returns:
            Random anime data or None if request fails
        """
        print("🎲 Fetching random anime...")
        
        anime_data = self._make_request("random/anime", use_cache=False)
        
        if anime_data is None:
            return None
        
        processed = self._process_anime_data(anime_data)
        print(f"✅ Random anime: {processed['title']}")
        return processed
    
    def get_top_anime(self, page: int = 1, limit: int = 25) -> List[Dict]:
        """
        Get top-rated anime
        
        Args:
            page: Page number (25 per page)
            limit: Number of results per page (max 25)
            
        Returns:
            List of top-rated anime
        """
        print(f"📈 Fetching top anime (page {page})...")
        
        results = self._make_request(
            "top/anime",
            {"page": page, "limit": limit},
            use_cache=True
        )
        
        if not results:
            return []
        
        processed = [self._process_anime_data(anime) for anime in results]
        print(f"✅ Retrieved {len(processed)} top anime")
        return processed
    
    def clear_cache(self) -> None:
        """
        Clear all cached API responses
        Useful if you want fresh data
        """
        print("🗑️  Clearing cache...")
        
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🎌 Jikan API Wrapper - Testing")
    print("=" * 60)
    
    # Initialize the client
    client = AnimeAPIClient()
    
    # Test 1: Search for an anime by name
    print("\n🧪 Test 1: Search by name")
    print("-" * 60)
    anime = client.search_anime_by_name("Death Note")
    if anime:
        print(f"✅ Found: {anime['title']}")
        print(f"   Score: {anime['score']}")
        print(f"   Type: {anime['type']}")
        print(f"   Episodes: {anime['episodes']}")
        print(f"   Genres: {', '.join(anime['genres'])}")
        print(f"   Studios: {', '.join(anime['studios'])}")
    
    # Test 2: Get autocomplete suggestions
    print("\n🧪 Test 2: Get suggestions")
    print("-" * 60)
    suggestions = client.get_anime_suggestions("Death", limit=5)
    for s in suggestions:
        print(f"  • {s['title']} ({s['year']}) - {s['score']}/10")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)