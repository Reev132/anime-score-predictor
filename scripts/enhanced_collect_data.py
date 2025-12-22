import requests
import pandas as pd
import time
import random
from datetime import datetime
import os
from typing import List, Dict, Optional


class MegaAnimeCollector:
    """
    MEGA anime data collector - designed to collect 3000+ anime
    More aggressive collection strategy with better coverage
    """

    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
        self.anime_list = []
        self.collected_ids = set()
        self.request_count = 0

        # Rate limiting (respectful but efficient)
        self.requests_per_minute = 60
        self.request_delay = 60 / self.requests_per_minute

    def safe_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Make a safe API request with error handling and rate limiting
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting with jitter
                time.sleep(self.request_delay + random.uniform(0, 0.3))
                self.request_count += 1

                if self.request_count % 100 == 0:
                    print(f"🔄 Made {self.request_count} requests | Collected {len(self.anime_list)} anime so far...")

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

    def collect_top_anime(self, pages: int = 20) -> None:
        """
        Collect top-rated anime (MORE PAGES!)
        """
        print(f"📈 Collecting top-rated anime ({pages} pages = ~{pages * 25} anime)...")
        for page in range(1, pages + 1):
            url = f"{self.base_url}/top/anime?page={page}&limit=25"
            data = self.safe_request(url)
            if not data or 'data' not in data:
                continue
            for anime in data['data']:
                if anime['mal_id'] not in self.collected_ids:
                    self.anime_list.append(self.extract_anime_info(anime))
                    self.collected_ids.add(anime['mal_id'])
            print(f"  ✅ Page {page}/{pages} - Total: {len(self.anime_list)} anime")

    def collect_seasonal_anime(self, years: List[int], seasons: List[str]) -> None:
        """
        Collect seasonal anime (MORE YEARS, MORE PER SEASON!)
        """
        print(f"🌸 Collecting seasonal anime from {len(years)} years...")
        target_per_season = 30  # Increased from 20
        
        for year in years:
            for season in seasons:
                url = f"{self.base_url}/seasons/{year}/{season}"
                data = self.safe_request(url)
                if not data or 'data' not in data:
                    continue
                    
                anime_data = data['data']
                season_count = 0
                
                for anime in anime_data[:target_per_season]:
                    if anime['mal_id'] not in self.collected_ids and anime.get('score') is not None:
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        season_count += 1
                        
                if season_count > 0:
                    print(f"  ✅ {year} {season}: +{season_count} anime (Total: {len(self.anime_list)})")

    def collect_by_genre(self, genres: List[int], limit_per_genre: int = 100) -> None:
        """
        Collect anime by genre (MORE PER GENRE!)
        """
        print(f"🏷️ Collecting anime by {len(genres)} genres...")
        genre_names = {
            1: "Action", 2: "Adventure", 4: "Comedy", 8: "Drama",
            10: "Fantasy", 14: "Horror", 22: "Romance", 24: "Sci-Fi",
            36: "Slice of Life", 37: "Supernatural", 41: "Suspense",
            7: "Mystery", 30: "Sports", 27: "Shounen", 25: "Shoujo"
        }
        
        for genre_id in genres:
            genre_name = genre_names.get(genre_id, f"Genre_{genre_id}")
            print(f"  🎭 Collecting {genre_name} anime...")
            collected_for_genre = 0
            page = 1
            
            while collected_for_genre < limit_per_genre and page <= 10:
                url = f"{self.base_url}/anime?genres={genre_id}&page={page}&limit=25&order_by=members&sort=desc"
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
                        
                if page_additions > 0:
                    print(f"    📄 Page {page}: +{page_additions} (Genre total: {collected_for_genre})")
                page += 1
                
            print(f"  ✅ {genre_name}: {collected_for_genre} anime")

    def collect_by_popularity_tiers(self) -> None:
        """
        NEW: Collect anime from different popularity tiers
        """
        print(f"🌟 Collecting anime from popularity tiers...")
        
        tiers = [
            (1, 500, "Top Tier"),
            (501, 2000, "High Popularity"),
            (2001, 5000, "Medium Popularity"),
            (5001, 10000, "Lower Popularity")
        ]
        
        for min_pop, max_pop, tier_name in tiers:
            print(f"  📊 {tier_name} (rank {min_pop}-{max_pop})...")
            collected = 0
            target = 100
            page = 1
            
            while collected < target and page <= 8:
                url = f"{self.base_url}/anime?page={page}&limit=25&order_by=popularity&sort=asc"
                data = self.safe_request(url)
                
                if not data or 'data' not in data:
                    break
                    
                for anime in data['data']:
                    popularity = anime.get('popularity', 999999)
                    if (min_pop <= popularity <= max_pop and
                            anime['mal_id'] not in self.collected_ids and
                            anime.get('score') is not None):
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        collected += 1
                        if collected >= target:
                            break
                page += 1
                
            print(f"    ✅ Collected {collected} from {tier_name}")

    def collect_by_type(self) -> None:
        """
        NEW: Ensure good coverage of all anime types
        """
        print(f"📺 Collecting diverse anime types...")
        
        types_targets = {
            'tv': 200,
            'movie': 150,
            'ova': 100,
            'ona': 80,
            'special': 50
        }
        
        for anime_type, target in types_targets.items():
            print(f"  🎬 Collecting {anime_type.upper()} anime...")
            collected = 0
            page = 1
            
            while collected < target and page <= 15:
                url = f"{self.base_url}/anime?type={anime_type}&page={page}&limit=25&order_by=score&sort=desc"
                data = self.safe_request(url)
                
                if not data or 'data' not in data:
                    break
                    
                for anime in data['data']:
                    if (anime['mal_id'] not in self.collected_ids and
                            anime.get('score') is not None):
                        self.anime_list.append(self.extract_anime_info(anime))
                        self.collected_ids.add(anime['mal_id'])
                        collected += 1
                        if collected >= target:
                            break
                page += 1
                
            print(f"    ✅ Collected {collected} {anime_type.upper()} anime")

    def collect_score_ranges(self, score_ranges: List[tuple]) -> None:
        """
        Collect anime from different score ranges (BALANCED DATASET!)
        """
        print(f"📊 Collecting anime from {len(score_ranges)} score ranges...")
        
        for min_score, max_score in score_ranges:
            print(f"  🎯 Score range: {min_score} - {max_score}")
            collected_in_range = 0
            page = 1
            target_per_range = 150  # Increased from 100
            
            while collected_in_range < target_per_range and page <= 15:
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
                            
                if page_additions > 0:
                    print(f"    📄 Page {page}: +{page_additions}")
                page += 1
                
            print(f"  ✅ Range {min_score}-{max_score}: {collected_in_range} anime")

    def collect_random_anime(self, count: int = 200) -> None:
        """
        Collect random anime for extra diversity (INCREASED!)
        """
        print(f"🎲 Collecting {count} random anime for diversity...")
        added = 0
        attempts = 0
        
        while added < count and attempts < count * 3:
            attempts += 1
            url = f"{self.base_url}/random/anime"
            data = self.safe_request(url)
            
            if not data or 'data' not in data:
                continue
                
            anime = data['data']
            if anime['mal_id'] not in self.collected_ids and anime.get('score') is not None:
                self.anime_list.append(self.extract_anime_info(anime))
                self.collected_ids.add(anime['mal_id'])
                added += 1
                
                if added % 25 == 0:
                    print(f"  ✅ Collected {added}/{count} random anime")
                    
        print(f"🎉 Finished collecting {added} random anime")

    def save_dataset(self, filename: str = None) -> str:
        """
        Save the collected dataset
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mega_anime_dataset_{timestamp}.csv"
            
        os.makedirs('data/raw', exist_ok=True)
        df = pd.DataFrame(self.anime_list).drop_duplicates(subset=['mal_id'])
        df = df.sort_values('score', ascending=False, na_position="last")
        df.to_csv(filename, index=False)
        
        print(f"\n🎉 Dataset Collection Complete!")
        print(f"📁 Saved as: {filename}")
        print(f"📊 Total anime: {len(df)}")
        
        if not df['score'].dropna().empty:
            print(f"⭐ Score range: {df['score'].dropna().min():.2f} - {df['score'].dropna().max():.2f}")
            print(f"📈 Average score: {df['score'].dropna().mean():.2f}")
        
        # Score distribution
        if len(df) > 0:
            print(f"\n📊 Score Distribution:")
            print(f"  • 9.0+: {len(df[df['score'] >= 9.0])} anime")
            print(f"  • 8.0-8.9: {len(df[(df['score'] >= 8.0) & (df['score'] < 9.0)])} anime")
            print(f"  • 7.0-7.9: {len(df[(df['score'] >= 7.0) & (df['score'] < 8.0)])} anime")
            print(f"  • 6.0-6.9: {len(df[(df['score'] >= 6.0) & (df['score'] < 7.0)])} anime")
            print(f"  • <6.0: {len(df[df['score'] < 6.0])} anime")
            
            # Type distribution
            print(f"\n📺 Type Distribution:")
            type_counts = df['type'].value_counts()
            for anime_type, count in type_counts.items():
                print(f"  • {anime_type}: {count} anime")
                
        return filename


def main():
    """
    MEGA data collection - target 3000+ anime!
    """
    print("🚀 Starting MEGA Anime Data Collection!")
    print("🎯 TARGET: 3000+ anime with scores")
    print("=" * 70)

    collector = MegaAnimeCollector()
    
    # Strategy 1: Top anime (500 anime)
    collector.collect_top_anime(pages=20)
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 2: Seasonal anime (540 anime from 9 years × 4 seasons × 15 each)
    years = list(range(2010, 2025))  # Extended range!
    seasons = ['winter', 'spring', 'summer', 'fall']
    collector.collect_seasonal_anime(years, seasons)
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 3: Genre-based (1400 anime from 14 genres × 100 each)
    popular_genres = [1, 2, 4, 7, 8, 10, 14, 22, 24, 25, 27, 30, 36, 37, 41]
    collector.collect_by_genre(popular_genres, limit_per_genre=100)
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 4: Score ranges (750 anime from 5 ranges × 150 each)
    score_ranges = [
        (8.5, 10.0),
        (7.5, 8.4),
        (6.5, 7.4),
        (5.5, 6.4),
        (0.0, 5.4)
    ]
    collector.collect_score_ranges(score_ranges)
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 5: Popularity tiers (400 anime)
    collector.collect_by_popularity_tiers()
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 6: Type diversity (580 anime)
    collector.collect_by_type()
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")
    
    # Strategy 7: Random for extra diversity (200 anime)
    collector.collect_random_anime(count=200)
    print(f"\n📊 Progress: {len(collector.anime_list)} anime collected\n")

    # Save final dataset
    filename = collector.save_dataset()
    df = pd.read_csv(filename)

    print(f"\n{'=' * 70}")
    print(f"🎉 MEGA COLLECTION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"🎯 Target: 3000+ anime")
    print(f"✅ Achieved: {len(df)} unique anime")
    print(f"📊 API Requests Made: {collector.request_count}")
    print(f"⏱️  Estimated time: ~{collector.request_count * 1.2 / 60:.0f} minutes")
    print(f"\n🚀 Ready for machine learning training!")
    print(f"{'=' * 70}\n")
    
    return df


if __name__ == "__main__":
    dataset = main()