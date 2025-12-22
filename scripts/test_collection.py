import requests
import json
import time

def test_jikan_api():
    """Simple test to make sure the API works"""
    print("Testing Jikan API connection...")

    # test with a popular anime (Attack on Titan)
    url = "https://api.jikan.moe/v4/anime/16498"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()['data']

        print("API connection successful!")
        print(f"Anime: {data['title']}")
        print(f"Score: {data['score']}")
        print(f"Rank: #{data['rank']}")
        print(f"Genres: {[g['name'] for g in data['genres']]}")

        return True

    # in case if API fails
    except Exception as e:
        print(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    test_jikan_api()