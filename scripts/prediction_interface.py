import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime

class AnimeScorePredictionInterface:
    """
    Easy-to-use interface for predicting anime scores
    """
    
    def __init__(self, model_path=None):
        """
        Load the trained model
        """
        if model_path is None:
            # Find the most recent model
            model_files = glob.glob("models/*.joblib")
            if not model_files:
                raise FileNotFoundError("❌ No trained models found in models/ directory")
            model_path = max(model_files, key=os.path.getctime)
        
        print(f"📁 Loading model from: {model_path}")
        
        # Load model package
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.model_name = self.model_package['model_name']
        self.scaler = self.model_package['scaler']
        self.feature_columns = self.model_package['feature_columns']
        self.model_results = self.model_package['model_results']
        
        print(f"🤖 Loaded model: {self.model_name}")
        print(f"📊 Model performance: RMSE = {self.model_results['test_rmse']:.3f}, R² = {self.model_results['test_r2']:.3f}")
        print(f"🔢 Expected features: {len(self.feature_columns)}")
    
    def create_prediction_template(self):
        """
        Create a template showing what features are needed
        """
        print("\n📝 Anime Prediction Template:")
        print("=" * 50)
        
        template = {
            # Basic info
            'title': 'My New Anime',
            'type': 'TV',  # TV, Movie, OVA, ONA, Special
            'episodes': 12,
            'year': 2024,
            'season': 'spring',  # spring, summer, fall, winter
            'source': 'Manga',  # Manga, Light novel, Original, etc.
            'status': 'Finished Airing',  # Currently Airing, Finished Airing
            'rating': 'PG-13',  # G, PG, PG-13, R, etc.
            
            # Popularity metrics (estimated)
            'members': 100000,  # Estimated number of members
            'favorites': 5000,  # Estimated favorites
            'scored_by': 50000,  # Estimated number of ratings
            'popularity': 500,  # Popularity rank (lower = more popular)
            
            # Content (genres as list)
            'genres': ['Action', 'Adventure', 'Fantasy'],  # List of genres
            'studios': ['Studio Bones'],  # List of studios
        }
        
        for key, value in template.items():
            print(f"  {key}: {value}")
        
        return template
    
    def predict_score(self, anime_info):
        """
        Predict anime score from anime information
        """
        print(f"\n🎯 Predicting score for: {anime_info.get('title', 'Unknown Anime')}")
        
        # Create feature vector
        feature_vector = self._create_feature_vector(anime_info)
        
        # Make prediction
        if self.model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
            # Use scaled features for linear models
            feature_vector_scaled = self.scaler.transform([feature_vector])
            predicted_score = self.model.predict(feature_vector_scaled)[0]
        else:
            # Use unscaled features for tree models
            predicted_score = self.model.predict([feature_vector])[0]
        
        # Clip prediction to valid score range (0-10)
        predicted_score = np.clip(predicted_score, 0, 10)
        
        # Calculate confidence interval (rough estimate based on model RMSE)
        rmse = self.model_results['test_rmse']
        confidence_interval = (
            max(0, predicted_score - 1.96 * rmse),  # Lower bound
            min(10, predicted_score + 1.96 * rmse)  # Upper bound
        )
        
        # Create result dictionary
        result = {
            'predicted_score': round(predicted_score, 2),
            'confidence_interval': (round(confidence_interval[0], 2), round(confidence_interval[1], 2)),
            'model_used': self.model_name,
            'model_rmse': round(rmse, 3)
        }
        
        # Display prediction
        print(f"🎌 Predicted Score: {result['predicted_score']:.2f}")
        print(f"📊 95% Confidence Interval: {result['confidence_interval'][0]:.2f} - {result['confidence_interval'][1]:.2f}")
        print(f"🤖 Model: {result['model_used']} (RMSE: {result['model_rmse']})")
        
        # Score interpretation
        if predicted_score >= 8.5:
            print("✨ Interpretation: Excellent anime! Likely to be highly rated by viewers.")
        elif predicted_score >= 7.5:
            print("👍 Interpretation: Very good anime. Should be well-received.")
        elif predicted_score >= 6.5:
            print("😊 Interpretation: Good anime. Average to above-average reception expected.")
        elif predicted_score >= 5.5:
            print("😐 Interpretation: Mediocre anime. Mixed reception likely.")
        else:
            print("😬 Interpretation: Below average anime. May struggle with ratings.")
        
        return result
    
    def _create_feature_vector(self, anime_info):
        """
        Convert anime info dictionary to feature vector matching training data
        """
        # Initialize feature vector with zeros
        feature_vector = np.zeros(len(self.feature_columns))
        
        # Current year for age calculation
        current_year = datetime.now().year
        
        # Fill in features based on anime_info
        for i, feature in enumerate(self.feature_columns):
            
            # === NUMERICAL FEATURES ===
            if feature == 'episodes':
                feature_vector[i] = anime_info.get('episodes', 12)
            elif feature == 'year':
                feature_vector[i] = anime_info.get('year', current_year)
            elif feature == 'members':
                feature_vector[i] = anime_info.get('members') or 50000
            elif feature == 'favorites':
                feature_vector[i] = anime_info.get('favorites') or 2000
            elif feature == 'scored_by':
                feature_vector[i] = anime_info.get('scored_by') or 25000
            elif feature == 'popularity':
                feature_vector[i] = anime_info.get('popularity') or 1000
            elif feature == 'anime_age':
                year = anime_info.get('year', current_year)
                feature_vector[i] = current_year - year
            elif feature == 'is_recent':
                year = anime_info.get('year', current_year)
                feature_vector[i] = 1 if (current_year - year) <= 5 else 0
            elif feature == 'engagement_rate':
                members = anime_info.get('members')
                scored_by = anime_info.get('scored_by')

                # Handle None values (unrated / unreleased anime)
                if members is None or members <= 0:
                    members = 1
                if scored_by is None or scored_by < 0:
                    scored_by = 0

                feature_vector[i] = scored_by / members

            elif feature == 'source_encoded':
                source_mapping = {
                    'Manga': 1, 'Light novel': 2, 'Visual novel': 3, 'Novel': 4,
                    'Original': 5, 'Game': 6, 'Web manga': 7, 'Other': 0
                }
                source = anime_info.get('source', 'Other')
                feature_vector[i] = source_mapping.get(source, 0)
            
            # === GENRE FEATURES ===
            elif feature.startswith('genre_'):
                genre_name = feature.replace('genre_', '').replace('_', ' ').replace('-', '-')
                
                genres = anime_info.get('genres', [])
                if isinstance(genres, str):
                    genres = [g.strip() for g in genres.split(',')]
                
                # Check if this genre is present
                genre_present = False
                for genre in genres:
                    if (genre.replace(' ', '_').replace('-', '_').lower() == 
                        genre_name.replace(' ', '_').replace('-', '_').lower()):
                        genre_present = True
                        break
                
                feature_vector[i] = 1 if genre_present else 0
            
            # === STUDIO FEATURES ===
            elif feature.startswith('studio_'):
                studio_name = feature.replace('studio_', '').replace('_', ' ')
                studios = anime_info.get('studios', [])
                if isinstance(studios, str):
                    studios = [s.strip() for s in studios.split(',')]
                
                # Check if this studio is present
                studio_present = False
                for studio in studios:
                    if studio.replace(' ', '_').lower() == studio_name.replace(' ', '_').lower():
                        studio_present = True
                        break
                
                feature_vector[i] = 1 if studio_present else 0
            
            # === CATEGORICAL FEATURES (ONE-HOT ENCODED) ===
            elif feature.startswith('type_'):
                anime_type = anime_info.get('type', 'TV')
                expected_type = feature.replace('type_', '')
                feature_vector[i] = 1 if anime_type == expected_type else 0
            
            elif feature.startswith('status_'):
                status = anime_info.get('status', 'Finished Airing')
                expected_status = feature.replace('status_', '').replace('_', ' ')
                feature_vector[i] = 1 if status == expected_status else 0
            
            elif feature.startswith('rating_'):
                rating = anime_info.get('rating', 'PG-13')
                expected_rating = feature.replace('rating_', '').replace('_', '-')
                feature_vector[i] = 1 if rating == expected_rating else 0
            
            elif feature.startswith('season_'):
                season = anime_info.get('season', 'spring')
                expected_season = feature.replace('season_', '')
                feature_vector[i] = 1 if season == expected_season else 0
            
            elif feature.startswith('episode_category_'):
                episodes = anime_info.get('episodes', 12)
                category = feature.replace('episode_category_', '')
                
                if category == 'Movie_Special' and episodes <= 1:
                    feature_vector[i] = 1
                elif category == 'Short' and 2 <= episodes <= 12:
                    feature_vector[i] = 1
                elif category == 'Standard' and 13 <= episodes <= 26:
                    feature_vector[i] = 1
                elif category == 'Long' and 27 <= episodes <= 50:
                    feature_vector[i] = 1
                elif category == 'Very_Long' and episodes > 50:
                    feature_vector[i] = 1
            
            elif feature.startswith('popularity_tier_'):
                popularity = anime_info.get('popularity', 1000)
                tier = feature.replace('popularity_tier_', '')
                
                if tier == 'Top_Tier' and popularity <= 100:
                    feature_vector[i] = 1
                elif tier == 'High' and 101 <= popularity <= 1000:
                    feature_vector[i] = 1
                elif tier == 'Medium' and 1001 <= popularity <= 5000:
                    feature_vector[i] = 1
                elif tier == 'Low' and popularity > 5000:
                    feature_vector[i] = 1
        
        return feature_vector
    
    def batch_predict(self, anime_list):
        """
        Predict scores for multiple anime
        """
        results = []
        for anime_info in anime_list:
            result = self.predict_score(anime_info)
            result['title'] = anime_info.get('title', 'Unknown')
            results.append(result)
        
        return results
    
    def compare_anime(self, anime_list):
        """
        Compare predicted scores for multiple anime
        """
        print("\n🏆 Anime Score Comparison:")
        print("=" * 60)
        
        results = self.batch_predict(anime_list)
        
        # Sort by predicted score
        results_sorted = sorted(results, key=lambda x: x['predicted_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Title':<25} {'Predicted Score':<15} {'Confidence Interval':<20}")
        print("-" * 70)
        
        for i, result in enumerate(results_sorted, 1):
            title = result['title'][:24] + '...' if len(result['title']) > 24 else result['title']
            score = result['predicted_score']
            ci = f"{result['confidence_interval'][0]:.2f}-{result['confidence_interval'][1]:.2f}"
            
            print(f"{i:<4} {title:<25} {score:<15.2f} {ci:<20}")
        
        return results_sorted

def demo_predictions():
    """
    Demo function showing how to use the prediction interface
    """
    print("🚀 Anime Score Predictor Demo!")
    print("=" * 40)
    
    # Load the predictor
    try:
        predictor = AnimeScorePredictionInterface()
    except FileNotFoundError:
        print("❌ No trained model found. Please run the ML pipeline first!")
        return
    
    # Show template
    template = predictor.create_prediction_template()
    
    # Example predictions
    print("\n🎬 Example Predictions:")
    
    # Example 1: High-budget action anime
    anime1 = {
        'title': 'Epic Battle Saga',
        'type': 'TV',
        'episodes': 24,
        'year': 2024,
        'season': 'spring',
        'source': 'Manga',
        'status': 'Finished Airing',
        'rating': 'PG-13',
        'members': 500000,
        'favorites': 25000,
        'scored_by': 200000,
        'popularity': 50,
        'genres': ['Action', 'Adventure', 'Drama'],
        'studios': ['Madhouse']
    }
    
    # Example 2: Slice of life comedy
    anime2 = {
        'title': 'Daily Life Chronicles',
        'type': 'TV',
        'episodes': 12,
        'year': 2024,
        'season': 'summer',
        'source': 'Original',
        'status': 'Finished Airing',
        'rating': 'PG',
        'members': 150000,
        'favorites': 8000,
        'scored_by': 75000,
        'popularity': 200,
        'genres': ['Comedy', 'Slice of Life'],
        'studios': ['Kyoto Animation']
    }
    
    # Example 3: Low-budget ecchi anime
    anime3 = {
        'title': 'Fan Service Paradise',
        'type': 'TV',
        'episodes': 12,
        'year': 2024,
        'season': 'fall',
        'source': 'Light novel',
        'status': 'Finished Airing',
        'rating': 'R',
        'members': 80000,
        'favorites': 2000,
        'scored_by': 30000,
        'popularity': 1500,
        'genres': ['Ecchi', 'Comedy', 'School'],
        'studios': ['Studio Deen']
    }
    
    # Make predictions
    predictor.predict_score(anime1)
    predictor.predict_score(anime2)
    predictor.predict_score(anime3)
    
    # Compare them
    anime_comparison = [anime1, anime2, anime3]
    predictor.compare_anime(anime_comparison)
    
    return predictor

if __name__ == "__main__":
    demo_predictions()