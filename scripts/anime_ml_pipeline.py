import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import glob
import os

class AnimeScorePredictor:
    """
    Complete ML pipeline for predicting anime scores
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'score'
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, csv_file=None):
        """
        Load the most recent dataset
        """
        if csv_file is None:
            # Find the most recent CSV file
            csv_files = glob.glob("data/raw/*.csv")
            if not csv_files:
                raise FileNotFoundError("❌ No CSV files found in data/raw/")
            csv_file = max(csv_files, key=os.path.getctime)
        
        print(f"📁 Loading data from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        
        print(f"📊 Dataset loaded: {self.df.shape[0]} anime, {self.df.shape[1]} features")
        print(f"🎯 Score range: {self.df[self.target_column].min():.2f} - {self.df[self.target_column].max():.2f}")
        
        return self.df
    
    def preprocess_data(self):
        """
        Advanced preprocessing and feature engineering
        """
        print("🔧 Starting data preprocessing and feature engineering...")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Remove rows without scores (our target)
        df_processed = df_processed[df_processed[self.target_column].notna()]
        print(f"📊 After removing missing scores: {len(df_processed)} anime")
        
        # === NUMERICAL FEATURES ===
        numerical_features = ['episodes', 'year', 'members', 'favorites', 'scored_by', 'popularity', 'rank']
        
        # Handle missing numerical values
        for col in numerical_features:
            if col in df_processed.columns:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                print(f"  📈 Filled {col} missing values with median: {median_val}")
        
        # === FEATURE ENGINEERING ===
        print("⚡ Engineering new features...")
        
        # 1. Age of anime (how old is it?)
        current_year = datetime.now().year
        if 'year' in df_processed.columns:
            df_processed['anime_age'] = current_year - df_processed['year'].fillna(current_year)
            df_processed['is_recent'] = (df_processed['anime_age'] <= 5).astype(int)  # Released in last 5 years
        
        # 2. Episode categories
        if 'episodes' in df_processed.columns:
            df_processed['episode_category'] = pd.cut(
                df_processed['episodes'].fillna(0), 
                bins=[-1, 1, 12, 26, 50, 1000], 
                labels=['Movie/Special', 'Short', 'Standard', 'Long', 'Very_Long']
            ).astype(str)
        
        # 3. Popularity tiers
        if 'popularity' in df_processed.columns:
            df_processed['popularity_tier'] = pd.cut(
                df_processed['popularity'].fillna(df_processed['popularity'].max()), 
                bins=[0, 100, 1000, 5000, np.inf], 
                labels=['Top_Tier', 'High', 'Medium', 'Low']
            ).astype(str)
        
        # 4. Member engagement (members to scored_by ratio)
        if 'members' in df_processed.columns and 'scored_by' in df_processed.columns:
            df_processed['engagement_rate'] = df_processed['scored_by'] / (df_processed['members'] + 1)  # +1 to avoid division by zero
        
        # === CATEGORICAL FEATURES ===
        print("🏷️ Processing categorical features...")
        
        # 1. Basic categorical encoding
        categorical_cols = ['type', 'status', 'source', 'rating', 'season', 'episode_category', 'popularity_tier']
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # 2. Genre processing (one-hot encoding)
        if 'genres' in df_processed.columns:
            print("  🎭 Processing genres...")
            # Get all unique genres
            all_genres = set()
            for genres_str in df_processed['genres'].dropna():
                if isinstance(genres_str, str):
                    genres = [g.strip() for g in genres_str.split(',')]
                    all_genres.update(genres)
            
            # Create binary columns for each genre
            for genre in sorted(all_genres):
                if genre and genre != '':  # Skip empty genres
                    df_processed[f'genre_{genre.replace(" ", "_").replace("-", "_")}'] = \
                        df_processed['genres'].str.contains(genre, na=False).astype(int)
            
            print(f"    ✅ Created {len(all_genres)} genre features")
        
        # 3. Studio processing (encode top studios, group others)
        if 'studios' in df_processed.columns:
            print("  🏢 Processing studios...")
            # Get studio counts
            all_studios = []
            for studios_str in df_processed['studios'].dropna():
                if isinstance(studios_str, str):
                    studios = [s.strip() for s in studios_str.split(',')]
                    all_studios.extend(studios)
            
            studio_counts = pd.Series(all_studios).value_counts()
            top_studios = studio_counts.head(20).index.tolist()  # Top 20 studios
            
            # Create features for top studios
            for studio in top_studios:
                if studio and studio != '':
                    df_processed[f'studio_{studio.replace(" ", "_").replace("-", "_")}'] = \
                        df_processed['studios'].str.contains(studio, na=False, regex=False).astype(int)
            
            print(f"    ✅ Created {len(top_studios)} studio features")
        
        # 4. Source material encoding
        if 'source' in df_processed.columns:
            source_mapping = {
                'Manga': 1, 'Light novel': 2, 'Visual novel': 3, 'Novel': 4,
                'Original': 5, 'Game': 6, 'Web manga': 7, 'Other': 0
            }
            df_processed['source_encoded'] = df_processed['source'].map(source_mapping).fillna(0)
        
        # === FINAL FEATURE SELECTION ===
        # Select features for modeling
        feature_columns = []
        
        # Add numerical features
        numerical_features_final = ['episodes', 'year', 'members', 'favorites', 'scored_by', 
                                  'popularity', 'anime_age', 'is_recent', 'engagement_rate', 'source_encoded']
        for col in numerical_features_final:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        # Add genre features
        genre_cols = [col for col in df_processed.columns if col.startswith('genre_')]
        feature_columns.extend(genre_cols)
        
        # Add studio features
        studio_cols = [col for col in df_processed.columns if col.startswith('studio_')]
        feature_columns.extend(studio_cols)
        
        # Add categorical features (one-hot encoded)
        categorical_for_encoding = ['type', 'status', 'rating', 'season', 'episode_category', 'popularity_tier']
        for col in categorical_for_encoding:
            if col in df_processed.columns:
                # One-hot encode
                dummies = pd.get_dummies(df_processed[col], prefix=col, dummy_na=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                feature_columns.extend(dummies.columns.tolist())
        
        # Store processed data and feature columns
        self.df_processed = df_processed
        self.feature_columns = feature_columns
        
        print(f"✅ Preprocessing complete!")
        print(f"📊 Final dataset: {len(df_processed)} anime")
        print(f"🔢 Total features: {len(feature_columns)}")
        print(f"📈 Features include: numerical ({len([c for c in feature_columns if not c.startswith(('genre_', 'studio_', 'type_', 'status_', 'rating_', 'season_', 'episode_', 'popularity_'))])}), genres ({len(genre_cols)}), studios ({len(studio_cols)}), categorical ({len(feature_columns) - len([c for c in feature_columns if not c.startswith(('genre_', 'studio_', 'type_', 'status_', 'rating_', 'season_', 'episode_', 'popularity_'))]) - len(genre_cols) - len(studio_cols)})")
        
        return df_processed
    
    def prepare_training_data(self):
        """
        Prepare X and y for training
        """
        print("🎯 Preparing training data...")
        
        # Get features and target
        X = self.df_processed[self.feature_columns].copy()
        y = self.df_processed[self.target_column].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.X_train_df, self.X_test_df = X_train, X_test  # Keep unscaled for tree models
        
        print(f"📊 Training set: {len(X_train)} anime")
        print(f"📊 Test set: {len(X_test)} anime")
        print(f"🎯 Target distribution - Train: {y_train.mean():.2f}±{y_train.std():.2f}, Test: {y_test.mean():.2f}±{y_test.std():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """
        Train multiple models and compare performance
        """
        print("🤖 Training multiple models...")
        
        # Define models to try
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"  🔄 Training {name}...")
            
            # Use scaled data for linear models and SVR, unscaled for tree models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                X_train_use, X_test_use = self.X_train, self.X_test
            else:
                X_train_use, X_test_use = self.X_train_df.values, self.X_test_df.values
            
            # Train the model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_use)
            test_pred = model.predict(X_test_use)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            # Cross-validation score
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression']:
                cv_score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()
            else:
                cv_score = cross_val_score(model, self.X_train_df.values, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()
            cv_rmse = np.sqrt(-cv_score)
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'test_predictions': test_pred
            }
            
            print(f"    ✅ {name}: Test RMSE = {test_rmse:.3f}, Test R² = {test_r2:.3f}, CV RMSE = {cv_rmse:.3f}")
        
        self.models = results
        
        # Find best model (based on cross-validation RMSE)
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_rmse'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name} (CV RMSE: {results[best_model_name]['cv_rmse']:.3f})")
        
        return results
    
    def evaluate_models(self):
        """
        Create comprehensive evaluation plots and metrics
        """
        print("📊 Creating model evaluation visualizations...")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. RMSE Comparison
        model_names = list(self.models.keys())
        train_rmses = [self.models[name]['train_rmse'] for name in model_names]
        test_rmses = [self.models[name]['test_rmse'] for name in model_names]
        cv_rmses = [self.models[name]['cv_rmse'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        axes[0,0].bar(x - width, train_rmses, width, label='Train RMSE', alpha=0.7)
        axes[0,0].bar(x, test_rmses, width, label='Test RMSE', alpha=0.7)
        axes[0,0].bar(x + width, cv_rmses, width, label='CV RMSE', alpha=0.7)
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. R² Comparison
        train_r2s = [self.models[name]['train_r2'] for name in model_names]
        test_r2s = [self.models[name]['test_r2'] for name in model_names]
        
        axes[0,1].bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.7)
        axes[0,1].bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.7)
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Best Model Predictions vs Actual
        best_pred = self.models[self.best_model_name]['test_predictions']
        axes[0,2].scatter(self.y_test, best_pred, alpha=0.6)
        axes[0,2].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0,2].set_xlabel('Actual Score')
        axes[0,2].set_ylabel('Predicted Score')
        axes[0,2].set_title(f'Best Model: {self.best_model_name}\nPredictions vs Actual')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Residuals plot for best model
        residuals = self.y_test - best_pred
        axes[1,0].scatter(best_pred, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Score')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title(f'{self.best_model_name}: Residuals Plot')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. MAE Comparison
        train_maes = [self.models[name]['train_mae'] for name in model_names]
        test_maes = [self.models[name]['test_mae'] for name in model_names]
        
        axes[1,1].bar(x - width/2, train_maes, width, label='Train MAE', alpha=0.7)
        axes[1,1].bar(x + width/2, test_maes, width, label='Test MAE', alpha=0.7)
        axes[1,1].set_xlabel('Models')
        axes[1,1].set_ylabel('Mean Absolute Error')
        axes[1,1].set_title('MAE Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            # Get top 15 most important features
            top_indices = np.argsort(feature_importance)[-15:]
            top_features = [self.feature_columns[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            axes[1,2].barh(range(len(top_features)), top_importance)
            axes[1,2].set_yticks(range(len(top_features)))
            axes[1,2].set_yticklabels(top_features)
            axes[1,2].set_xlabel('Feature Importance')
            axes[1,2].set_title(f'{self.best_model_name}: Top 15 Features')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, f'{self.best_model_name}\ndoes not provide\nfeature importances', 
                          ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
            axes[1,2].set_title('Feature Importance Not Available')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print(f"\n📈 Detailed Model Results:")
        print("="*80)
        for name, results in self.models.items():
            print(f"\n🤖 {name}:")
            print(f"  Train RMSE: {results['train_rmse']:.4f}")
            print(f"  Test RMSE:  {results['test_rmse']:.4f}")
            print(f"  CV RMSE:    {results['cv_rmse']:.4f}")
            print(f"  Train R²:   {results['train_r2']:.4f}")
            print(f"  Test R²:    {results['test_r2']:.4f}")
            print(f"  Train MAE:  {results['train_mae']:.4f}")
            print(f"  Test MAE:   {results['test_mae']:.4f}")
            
            # Overfitting check
            overfit_score = results['train_rmse'] / results['test_rmse']
            if overfit_score < 0.8:
                print(f"  📊 Status: Possibly underfitting (train/test RMSE ratio: {overfit_score:.3f})")
            elif overfit_score > 1.2:
                print(f"  ⚠️  Status: Possibly overfitting (train/test RMSE ratio: {overfit_score:.3f})")
            else:
                print(f"  ✅ Status: Good balance (train/test RMSE ratio: {overfit_score:.3f})")
    
    def save_model(self, filename=None):
        """
        Save the best model and preprocessing components
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/anime_predictor_{timestamp}.joblib"
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessing components
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_results': self.models[self.best_model_name]
        }
        
        joblib.dump(model_package, filename)
        print(f"💾 Best model ({self.best_model_name}) saved to: {filename}")
        
        return filename
    
    def predict_anime_score(self, anime_features):
        """
        Predict score for new anime
        """
        # This would be used for making predictions on new data
        # Implementation depends on how you want to input new anime features
        pass

def main():
    """
    Main ML pipeline execution
    """
    print("🤖 Starting Anime Score Prediction ML Pipeline!")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = AnimeScorePredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    df_processed = predictor.preprocess_data()
    
    # Prepare training data
    X_train, X_test, y_train, y_test = predictor.prepare_training_data()
    
    # Train models
    results = predictor.train_models()
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Save the best model
    model_file = predictor.save_model()
    
    print(f"\n🎉 ML Pipeline Complete!")
    print(f"🏆 Best model: {predictor.best_model_name}")
    print(f"📊 Test RMSE: {results[predictor.best_model_name]['test_rmse']:.3f}")
    print(f"📊 Test R²: {results[predictor.best_model_name]['test_r2']:.3f}")
    print(f"💾 Model saved: {model_file}")
    print(f"📈 Evaluation plots saved: model_evaluation.png")
    
    return predictor

if __name__ == "__main__":
    predictor = main()