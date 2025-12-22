import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def analyze_anime_dataset(csv_file):
    """
    Complete analysis of your anime dataset
    """
    print("Loading and analyzing your anime dataset...")
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    print(f"Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nScore Range: {df['score'].min():.2f} - {df['score'].max():.2f}")
    print(f"Average Score: {df['score'].mean():.2f}")
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Basic statistics
    print(f"\nScore Statistics:")
    print(df['score'].describe())
    
    # Create comprehensive visualizations
    create_visualizations(df)
    
    # Analyze genres
    analyze_genres(df)
    
    # Analyze studios
    analyze_studios(df)
    
    # Year and type analysis
    analyze_temporal_patterns(df)
    
    # Generate insights
    generate_insights(df)
    
    return df

def create_visualizations(df):
    """
    Create comprehensive visualizations
    """
    print("Creating visualizations...")
    
    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Anime Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Score Distribution
    axes[0,0].hist(df['score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0,0].set_title('Score Distribution')
    axes[0,0].set_xlabel('MAL Score')
    axes[0,0].set_ylabel('Count')
    axes[0,0].axvline(df['score'].mean(), color='red', linestyle='--', label=f'Mean: {df["score"].mean():.2f}')
    axes[0,0].legend()
    
    # 2. Type Distribution
    type_counts = df['type'].value_counts()
    axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Anime Types')
    
    # 3. Episodes vs Score
    df_episodes = df[df['episodes'].notna() & (df['episodes'] > 0)]
    scatter = axes[0,2].scatter(df_episodes['episodes'], df_episodes['score'], alpha=0.6, c=df_episodes['year'], cmap='viridis')
    axes[0,2].set_xlabel('Episodes')
    axes[0,2].set_ylabel('Score')
    axes[0,2].set_title('Episodes vs Score (colored by year)')
    plt.colorbar(scatter, ax=axes[0,2])
    
    # 4. Year Distribution
    year_counts = df['year'].value_counts().sort_index()
    axes[1,0].plot(year_counts.index, year_counts.values, marker='o')
    axes[1,0].set_title('Anime Count by Year')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Count')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Score vs Popularity
    df_pop = df[df['popularity'].notna()]
    axes[1,1].scatter(df_pop['popularity'], df_pop['score'], alpha=0.6)
    axes[1,1].set_xlabel('Popularity Rank (lower = more popular)')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Popularity vs Score')
    
    # 6. Members vs Score
    df_members = df[df['members'].notna()]
    axes[1,2].scatter(df_members['members'], df_members['score'], alpha=0.6)
    axes[1,2].set_xlabel('Members')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_title('Members vs Score')
    axes[1,2].set_xscale('log')
    
    # 7. Status Distribution
    status_counts = df['status'].value_counts()
    axes[2,0].bar(range(len(status_counts)), status_counts.values)
    axes[2,0].set_xticks(range(len(status_counts)))
    axes[2,0].set_xticklabels(status_counts.index, rotation=45)
    axes[2,0].set_title('Anime Status')
    axes[2,0].set_ylabel('Count')
    
    # 8. Score by Type (Box plot)
    df_type_score = df[df['type'].isin(['TV', 'Movie', 'OVA', 'ONA'])]
    sns.boxplot(data=df_type_score, x='type', y='score', ax=axes[2,1])
    axes[2,1].set_title('Score Distribution by Type')
    axes[2,1].tick_params(axis='x', rotation=45)
    
    # 9. Top 10 Studios by Count
    studio_counts = df['studios'].value_counts().head(10)
    axes[2,2].barh(range(len(studio_counts)), studio_counts.values)
    axes[2,2].set_yticks(range(len(studio_counts)))
    axes[2,2].set_yticklabels(studio_counts.index)
    axes[2,2].set_title('Top 10 Studios by Count')
    axes[2,2].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('anime_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Dashboard saved as 'anime_analysis_dashboard.png'")

def analyze_genres(df):
    """
    Deep dive into genre analysis
    """
    print("\n🏷️ Genre Analysis:")
    
    # Extract all genres
    all_genres = []
    for genres_str in df['genres'].dropna():
        if isinstance(genres_str, str):
            all_genres.extend([g.strip() for g in genres_str.split(',')])
    
    # Count genres
    genre_counts = pd.Series(all_genres).value_counts()
    print(f"📊 Top 15 Most Common Genres:")
    for i, (genre, count) in enumerate(genre_counts.head(15).items()):
        print(f"  {i+1:2d}. {genre}: {count} ({count/len(df)*100:.1f}%)")
    
    # Analyze genre performance
    print(f"\n⭐ Genre Performance Analysis:")
    genre_scores = {}
    for genre in genre_counts.head(10).index:
        genre_anime = df[df['genres'].str.contains(genre, na=False)]
        if len(genre_anime) >= 5:  # Only genres with at least 5 anime
            avg_score = genre_anime['score'].mean()
            genre_scores[genre] = {
                'count': len(genre_anime),
                'avg_score': avg_score,
                'median_score': genre_anime['score'].median()
            }
    
    # Sort by average score
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for genre, stats in sorted_genres:
        print(f"  {genre}: {stats['avg_score']:.2f} avg (n={stats['count']})")
    
    # Create genre performance chart
    plt.figure(figsize=(12, 8))
    genres = [item[0] for item in sorted_genres]
    scores = [item[1]['avg_score'] for item in sorted_genres]
    counts = [item[1]['count'] for item in sorted_genres]
    
    bars = plt.barh(genres, scores, color='lightcoral', alpha=0.7)
    plt.xlabel('Average Score')
    plt.title('Average Score by Genre (Top 10 genres with 5+ anime)')
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'n={count}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('genre_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return genre_scores

def analyze_studios(df):
    """
    Analyze studio performance
    """
    print(f"\n🏢 Studio Analysis:")
    
    # Count studios
    studio_counts = df['studios'].value_counts()
    print(f"📊 Top 10 Most Prolific Studios:")
    for i, (studio, count) in enumerate(studio_counts.head(10).items()):
        print(f"  {i+1:2d}. {studio}: {count} anime")
    
    # Studio performance (only studios with 3+ anime)
    print(f"\n⭐ Studio Performance (3+ anime):")
    studio_scores = {}
    for studio in studio_counts.index:
        if studio_counts[studio] >= 3:
            studio_anime = df[df['studios'] == studio]
            avg_score = studio_anime['score'].mean()
            studio_scores[studio] = {
                'count': len(studio_anime),
                'avg_score': avg_score,
                'best_anime': studio_anime.loc[studio_anime['score'].idxmax(), 'title']
            }
    
    # Sort by average score
    sorted_studios = sorted(studio_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for studio, stats in sorted_studios[:15]:
        print(f"  {studio}: {stats['avg_score']:.2f} avg (n={stats['count']}) - Best: {stats['best_anime']}")

def analyze_temporal_patterns(df):
    """
    Analyze patterns over time
    """
    print(f"\n📅 Temporal Analysis:")
    
    # Score trends by year
    df_year = df[df['year'].notna()].copy()
    year_stats = df_year.groupby('year')['score'].agg(['mean', 'count']).reset_index()
    
    # Only years with at least 3 anime
    year_stats = year_stats[year_stats['count'] >= 3]
    
    print(f"📈 Average Score by Year (3+ anime):")
    for _, row in year_stats.sort_values('mean', ascending=False).head(10).iterrows():
        print(f"  {int(row['year'])}: {row['mean']:.2f} avg (n={row['count']})")
    
    # Episode count analysis
    print(f"\n📺 Episode Count Analysis:")
    df_episodes = df[df['episodes'].notna() & (df['episodes'] > 0)].copy()
    
    # Create episode bins
    df_episodes['episode_category'] = pd.cut(df_episodes['episodes'], 
                                           bins=[0, 1, 12, 26, 50, 1000], 
                                           labels=['Movie/Special', 'Short (2-12)', 'Standard (13-26)', 'Long (27-50)', 'Very Long (50+)'])
    
    episode_stats = df_episodes.groupby('episode_category')['score'].agg(['mean', 'count'])
    
    for category, row in episode_stats.iterrows():
        print(f"  {category}: {row['mean']:.2f} avg (n={row['count']})")

def generate_insights(df):
    """
    Generate key insights from the data
    """
    print(f"\n🔍 Key Insights:")
    
    # Correlation analysis
    numeric_cols = ['score', 'rank', 'popularity', 'members', 'episodes', 'year']
    correlation_data = df[numeric_cols].corr()['score'].sort_values(key=abs, ascending=False)
    
    print(f"📊 Correlations with Score:")
    for col, corr in correlation_data.items():
        if col != 'score' and not pd.isna(corr):
            direction = "📈 Positive" if corr > 0 else "📉 Negative"
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            print(f"  {col}: {corr:.3f} ({direction}, {strength})")
    
    # Identify patterns
    print(f"\n🎯 Interesting Patterns:")
    
    # High scoring anime characteristics
    top_anime = df.nlargest(20, 'score')
    
    # Most common types in top anime
    top_types = top_anime['type'].value_counts()
    print(f"  • Top 20 anime types: {dict(top_types)}")
    
    # Most common genres in top anime
    top_genres = []
    for genres_str in top_anime['genres'].dropna():
        if isinstance(genres_str, str):
            top_genres.extend([g.strip() for g in genres_str.split(',')])
    
    top_genre_counts = pd.Series(top_genres).value_counts().head(5)
    print(f"  • Most common genres in top 20: {dict(top_genre_counts)}")
    
    # Studios with highest average scores
    studio_avg = df.groupby('studios')['score'].agg(['mean', 'count'])
    studio_avg = studio_avg[studio_avg['count'] >= 2]  # At least 2 anime
    top_studios = studio_avg.sort_values('mean', ascending=False).head(5)
    
    print(f"  • Top studios (2+ anime): {top_studios['mean'].to_dict()}")
    
    # Score distribution insights
    high_score_count = len(df[df['score'] >= 8.5])
    print(f"  • {high_score_count} anime ({high_score_count/len(df)*100:.1f}%) score 8.5+")
    print(f"  • Average episode count: {df['episodes'].mean():.1f}")
    print(f"  • Most popular type: {df['type'].mode().iloc[0]}")
    
    print(f"\n✨ Your dataset is ready for machine learning!")
    print(f"🎯 Next step: Build your prediction model!")

def main():
    """
    Main analysis function
    """
    # Look for the most recent CSV file
    import glob
    import os
    
    csv_files = glob.glob("data/raw/*.csv")
    if not csv_files:
        print("❌ No CSV files found in data/raw/")
        print("Make sure you've run collect_starter_data.py first!")
        return
    
    # Use the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"📁 Using file: {latest_file}")
    
    # Run the analysis
    df = analyze_anime_dataset(latest_file)
    
    print(f"\n🚀 Analysis complete! Check the generated charts!")
    print(f"📊 Dashboard saved as 'anime_analysis_dashboard.png'")
    print(f"🏷️ Genre analysis saved as 'genre_performance.png'")
    
    return df

if __name__ == "__main__":
    df = main()