import requests
import pandas as pd
import time
import json
from datetime import datetime

def collect_top_anime_simple(num_anime=250):
    """
    Collect top anime data - simple version to get started
    """
    print(f"🎌 Starting collection of top {num_anime} anime...")

    anime_list = []
    base_url = "https://api.jikan.moe/v4"

    # Calculate how many pages we need (25 anime per page)
    pages_needed = (num_anime + 24) // 25

    for page in range(1, pages_needed + 1):
        print(f"📄 Collecting page {page}/{pages_needed}")

        # Get top anime for this page
        url = f"{base_url}/top/anime?page={page}&limit=25"

        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()['data']

            for anime in data[:min(25, num_anime - len(anime_list))]:
                anime_info = {
                    'mal_id': anime.get('mal_id'),
                    'title': anime.get('title'),
                    'score': anime.get('score'),
                    'rank': anime.get('rank'),
                    'popularity': anime.get('popularity'),
                    'members': anime.get('members'),
                    'episodes': anime.get('episodes'),
                    'year': anime.get('year'),
                    'type': anime.get('type'),
                    'status': anime.get('status'),
                    'genres': ', '.join([g['name'] for g in anime.get('genres', [])]),
                    'studios': ', '.join([s['name'] for s in anime.get('studios', [])]),
                    'source': anime.get('source')
                }

                anime_list.append(anime_info)
                print(f"✅ {anime_info['title']} (Score: {anime_info['score']})")

                if len(anime_list) >= num_anime:
                    break

            # Rate limiting - wait 1 second between requests
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(anime_list)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/top_anime_{timestamp}.csv"

    # Create directory if it doesn't exist
    import os
    os.makedirs('data/raw', exist_ok=True)

    df.to_csv(filename, index=False)

    print(f"\n🎉 Collection complete!")
    print(f"📊 Collected {len(df)} anime")
    print(f"💾 Saved as: {filename}")

    # Show basic stats
    print(f"\n📈 Quick Stats:")
    print(f"Average Score: {df['score'].mean():.2f}")
    print(f"Score Range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"Most Common Type: {df['type'].mode().values[0]}")

    return df

if __name__ == "__main__":
    # Start with just 50 anime to test everything works
    df = collect_top_anime_simple(250)
    print("\n🚀 Ready for analysis! Check your data/ folder!")