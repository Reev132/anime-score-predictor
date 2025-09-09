import requests
import pandas as pd
import time
import random
from datetime import datetime
import os
from typing import List, Dict, Optional


class EnhancedAnimeCollector:
    """
    Enhanced anime data collector for diverse, robust dataset
    """

    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
        self.anime_list = []
        self.collected_ids = set()
        self.request_count = 0

        # Rate limiting
        self.requests_per_minute = 60
        self.request_delay = 60 / self.requests_per_minute  # ~1 sec per request

    def safe_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Make a safe API request with error handling and rate limiting
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting with jitter
                time.sleep(self.request_delay + random.uniform(0, 0.5))
                self.request_count += 1

                if self.request_count % 50 == 0:
                    print(f"🔄 Made {self.request_count} requests so far...")

                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"🕐 Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts")
                    return None
        return None

    def extract_anime_info(self, anime: Dict) -> Dict:
        """
        Extract and clean anime information
        """
        return {
            'mal_id': anime.get('mal_id'),
            'title': anime.get('title', ''),
            'title_english': anime.get('title_english', ''),
            'title_japanese': anime.get('title_japanese', ''),
            'score': anime.get('score'),
            'scored_by': anime.get('scored_by'),
            'rank': anime.get('rank'),
            'popularity': anime.get('popularity'),
            'members': anime.get('members'),
            'favorites': anime.get('favorites'),
            'episodes': anime.get('episodes'),
            'duration': anime.get('duration', ''),
            'year': anime.get('year'),
            'season': anime.get('season', ''),
            'type': anime.get('type'),
            'status': anime.get('status'),
            'rating': anime.get('rating', ''),
            'source': anime.get('source', ''),
            'genres': ', '.join([g['name'] for g in anime.get('genres', [])]),
            'themes': ', '.join([t['name'] for t in anime.get('themes', [])]),
            'demographics': ', '.join([d['name'] for d in anime.get('demographics', [])]),
            'studios': ', '.join([s['name'] for s in anime.get('studios', [])]),
            'producers': ', '.join([p['name'] for p in anime.get('producers', [])]),
            'licensors': ', '.join([l['name'] for l in anime.get('licensors', [])]),
            'synopsis': anime.get('synopsis', '')[:500] if anime.get('synopsis') else '',
            'background': anime.get('background', '')[:200] if anime.get('background') else ''
        }

    def collect_top_anime(self, pages: int = 10) -> None:
        print(f"📈 Collecting top-rated anime ({pages} pages)...")
        for page in range(1, pages + 1):
            url = f"{self.base_url}/top/anime?page={page}&limit=25"
            data = self.safe_request(url)
            if not data or 'data' not in data:
                continue
            for anime in data['data']:
                if anime['mal_id'] not in self.collected_ids:
                    self.anime_list.append(self.extract_anime_info(anime))
                    self.collected_ids.add(anime['mal_id'])
            print(f"  ✅ Page {page}/{pages} - Total collected: {len(self.anime_list)}")

    def collect_seasonal_anime(self, years: List[int], seasons: List[str]) -> None:
        print(f"🌸 Collecting seasonal anime from {len(years)} years...")
        collected_seasonal = 0
        target_per_season = 20
        for year in years:
            for season in seasons:
                if collected_seasonal >= 200:
                    break
                url = f"{self.base_url}/seasons/{year}/{season}"
                data = self.safe_request(url)
                if not data or 'data' not in data:
                    continue
                anime_data = data['data']
                random.shuffle(anime_data)
                season_count = 0
                for anime in anime_data[:target_per_season]:
                    if (anime['mal_id'] not in self.collected_ids and anime.get('score') is not None):
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        season_count += 1
                        collected_seasonal += 1
                print(f"  ✅ {year} {season}: +{season_count} anime")

    def collect_by_genre(self, genres: List[int], limit_per_genre: int = 50) -> None:
        print(f"🏷️ Collecting anime by {len(genres)} genres...")
        genre_names = {
            1: "Action", 2: "Adventure", 4: "Comedy", 8: "Drama",
            10: "Fantasy", 14: "Horror", 22: "Romance", 24: "Sci-Fi",
            36: "Slice of Life", 37: "Supernatural", 41: "Suspense"
        }
        for genre_id in genres:
            genre_name = genre_names.get(genre_id, f"Genre_{genre_id}")
            print(f"  🎭 Collecting {genre_name} anime...")
            collected_for_genre = 0
            page = 1
            while collected_for_genre < limit_per_genre and page <= 5:
                url = f"{self.base_url}/anime?genres={genre_id}&page={page}&limit=25&order_by=score&sort=desc"
                data = self.safe_request(url)
                if not data or 'data' not in data or not data['data']:
                    break
                page_additions = 0
                for anime in data['data']:
                    if (anime['mal_id'] not in self.collected_ids and
                            anime.get('score') is not None and
                            collected_for_genre < limit_per_genre):
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        collected_for_genre += 1
                        page_additions += 1
                print(f"    📄 Page {page}: +{page_additions} anime")
                page += 1
            print(f"  ✅ {genre_name}: {collected_for_genre} total anime")

    def collect_score_ranges(self, score_ranges: List[tuple]) -> None:
        print(f"📊 Collecting anime from {len(score_ranges)} score ranges...")
        for min_score, max_score in score_ranges:
            print(f"  🎯 Score range: {min_score} - {max_score}")
            collected_in_range = 0
            page = 1
            target_per_range = 100
            while collected_in_range < target_per_range and page <= 10:
                url = f"{self.base_url}/anime?min_score={min_score}&max_score={max_score}&page={page}&limit=25"
                data = self.safe_request(url)
                if not data or 'data' not in data or not data['data']:
                    break
                page_additions = 0
                for anime in data['data']:
                    if (anime['mal_id'] not in self.collected_ids and
                            anime.get('score') is not None and
                            min_score <= anime.get('score', 0) <= max_score):
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        collected_in_range += 1
                        page_additions += 1
                        if collected_in_range >= target_per_range:
                            break
                print(f"    📄 Page {page}: +{page_additions} anime")
                page += 1
            print(f"  ✅ Range {min_score}-{max_score}: {collected_in_range} anime")

    def collect_random_anime(self, count: int = 50) -> None:
        """
        Collect random anime from Jikan API for extra diversity
        """
        print(f"🎲 Collecting {count} random anime...")
        added = 0
        attempts = 0
        while added < count and attempts < count * 3:  # Prevent infinite loops
            attempts += 1
            url = f"{self.base_url}/random/anime"
            data = self.safe_request(url)
            if not data or 'data' not in data:
                continue
            anime = data['data']
            if anime['mal_id'] not in self.collected_ids:
                self.anime_list.append(self.extract_anime_info(anime))
                self.collected_ids.add(anime['mal_id'])
                added += 1
                if added % 10 == 0:
                    print(f"  ✅ Collected {added}/{count} random anime")
        print(f"🎉 Finished collecting {added} random anime")

    def save_dataset(self, filename: str = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/enhanced_anime_dataset_{timestamp}.csv"
        os.makedirs('data/raw', exist_ok=True)
        df = pd.DataFrame(self.anime_list).drop_duplicates(subset=['mal_id'])
        df = df.sort_values('score', ascending=False, na_position="last")
        df.to_csv(filename, index=False)
        print(f"\n🎉 Dataset Collection Complete!")
        print(f"📁 Saved as: {filename}")
        print(f"📊 Total anime: {len(df)}")
        if not df['score'].dropna().empty:
            print(f"⭐ Score range: {df['score'].dropna().min():.2f} - {df['score'].dropna().max():.2f}")
        else:
            print("⭐ Score range: No scores available")
        return filename


def main():
    print("🚀 Starting Enhanced Anime Data Collection!")
    print("=" * 50)

    collector = EnhancedAnimeCollector()
    collector.collect_top_anime(pages=8)
    years = list(range(2015, 2024))
    seasons = ['winter', 'spring', 'summer', 'fall']
    collector.collect_seasonal_anime(years, seasons)
    popular_genres = [1, 2, 4, 8, 10, 22, 24, 36, 37]
    collector.collect_by_genre(popular_genres, limit_per_genre=40)
    score_ranges = [(8.0, 10.0), (7.0, 7.9), (6.0, 6.9), (5.0, 5.9)]
    collector.collect_score_ranges(score_ranges)
    collector.collect_random_anime(count=100)  # 👈 new feature

    filename = collector.save_dataset()
    df = pd.read_csv(filename)

    if len(df) > 0:
        print(f"\n📈 Final Dataset Analysis:")
        print(f"🎯 Total unique anime: {len(df)}")
        print(f"📊 Score distribution:")
        print(f"  • 8.0+: {len(df[df['score'] >= 8.0])} anime ({len(df[df['score'] >= 8.0])/len(df)*100:.1f}%)")
        print(f"  • 7.0-7.9: {len(df[(df['score'] >= 7.0) & (df['score'] < 8.0)])} anime ({len(df[(df['score'] >= 7.0) & (df['score'] < 8.0)])/len(df)*100:.1f}%)")
        print(f"  • 6.0-6.9: {len(df[(df['score'] >= 6.0) & (df['score'] < 7.0)])} anime ({len(df[(df['score'] >= 6.0) & (df['score'] < 7.0)])/len(df)*100:.1f}%)")
        print(f"  • <6.0: {len(df[df['score'] < 6.0])} anime ({len(df[df['score'] < 6.0])/len(df)*100:.1f}%)")
    else:
        print("\n⚠️ Dataset is empty. No analysis available.")

    print(f"\n🎉 Your enhanced dataset is ready for machine learning!")
    return df


if __name__ == "__main__":
    dataset = main()
