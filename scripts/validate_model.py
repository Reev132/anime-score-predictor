import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from prediction_interface import AnimeScorePredictionInterface

class ModelValidator:
    """
    Comprehensive model validation and testing
    Tests model predictions against actual scores in the dataset
    """
    
    def __init__(self):
        """Initialize the validator"""
        print("🧪 Initializing Model Validator...")
        print("=" * 70)
        
        # Load the trained model
        try:
            self.predictor = AnimeScorePredictionInterface()
            print(f"✅ Model loaded: {self.predictor.model_name}")
            print(f"📊 Model RMSE: {self.predictor.model_results['test_rmse']:.3f}")
            print(f"📊 Model R²: {self.predictor.model_results['test_r2']:.3f}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # Load the dataset
        csv_files = glob.glob("data/raw/*.csv")
        if not csv_files:
            raise FileNotFoundError("❌ No CSV files found in data/raw/")
        
        self.dataset_path = max(csv_files, key=os.path.getctime)
        print(f"\n📁 Loading dataset: {self.dataset_path}")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"📊 Dataset loaded: {len(self.df)} anime")
        
        # Filter to only anime with actual scores
        self.df = self.df[self.df['score'].notna()].copy()
        print(f"📊 Anime with scores: {len(self.df)}")
        print("=" * 70)
    
    def prepare_anime_info(self, row):
        """
        Convert a dataframe row into the format expected by the predictor
        
        Args:
            row: A row from the anime dataframe
            
        Returns:
            Dictionary with anime information
        """
        # Handle genres (convert from comma-separated string to list)
        genres = []
        if pd.notna(row.get('genres')):
            genres = [g.strip() for g in str(row['genres']).split(',')]
        
        # Handle studios
        studios = []
        if pd.notna(row.get('studios')):
            studios = [s.strip() for s in str(row['studios']).split(',')]
        
        anime_info = {
            'title': row.get('title', 'Unknown'),
            'type': row.get('type', 'TV'),
            'episodes': row.get('episodes', 12),
            'year': row.get('year', 2020),
            'season': row.get('season', 'Unknown'),
            'source': row.get('source', 'Unknown'),
            'status': row.get('status', 'Finished Airing'),
            'rating': row.get('rating', 'PG-13'),
            'members': row.get('members', 50000),
            'favorites': row.get('favorites', 2000),
            'scored_by': row.get('scored_by', 25000),
            'popularity': row.get('popularity', 1000),
            'genres': genres,
            'studios': studios,
        }
        
        return anime_info
    
    def test_all_anime(self):
        """
        Test predictions on all anime in the dataset
        
        Returns:
            DataFrame with results
        """
        print("\n🧪 Testing model on all anime...")
        print("=" * 70)
        
        results = []
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"📊 Progress: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")
            
            try:
                # Prepare anime info
                anime_info = self.prepare_anime_info(row)
                
                # Make prediction (suppress output)
                import io
                import contextlib
                
                # Capture stdout to suppress prediction output
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    prediction = self.predictor.predict_score(anime_info)
                
                # Get actual score
                actual_score = row['score']
                predicted_score = prediction['predicted_score']
                ci_low, ci_high = prediction['confidence_interval']
                
                # Calculate error
                error = predicted_score - actual_score
                absolute_error = abs(error)
                
                # Check if actual score is within confidence interval
                within_ci = ci_low <= actual_score <= ci_high
                
                # Store results
                results.append({
                    'title': row.get('title', 'Unknown'),
                    'type': row.get('type', 'Unknown'),
                    'year': row.get('year', 'Unknown'),
                    'actual_score': actual_score,
                    'predicted_score': predicted_score,
                    'confidence_low': ci_low,
                    'confidence_high': ci_high,
                    'confidence_range': f"{ci_low:.2f} - {ci_high:.2f}",
                    'error': error,
                    'absolute_error': absolute_error,
                    'within_confidence_interval': within_ci,
                    'genres': row.get('genres', ''),
                    'studios': row.get('studios', ''),
                })
                
            except Exception as e:
                print(f"⚠️  Error predicting {row.get('title', 'Unknown')}: {e}")
                continue
        
        print(f"✅ Completed: {len(results)}/{total} predictions")
        print("=" * 70)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def calculate_metrics(self):
        """
        Calculate comprehensive accuracy metrics
        
        Returns:
            Dictionary with metrics
        """
        print("\n📊 Calculating Accuracy Metrics...")
        print("=" * 70)
        
        df = self.results_df
        
        # Basic metrics
        mae = df['absolute_error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        mape = (df['absolute_error'] / df['actual_score'] * 100).mean()
        
        # R² score
        ss_res = ((df['actual_score'] - df['predicted_score']) ** 2).sum()
        ss_tot = ((df['actual_score'] - df['actual_score'].mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        # Accuracy bands
        within_0_5 = len(df[df['absolute_error'] <= 0.5]) / len(df) * 100
        within_1_0 = len(df[df['absolute_error'] <= 1.0]) / len(df) * 100
        within_1_5 = len(df[df['absolute_error'] <= 1.5]) / len(df) * 100
        within_ci = len(df[df['within_confidence_interval']]) / len(df) * 100
        
        # Direction accuracy (did we predict higher/lower correctly?)
        mean_score = df['actual_score'].mean()
        df['actual_above_mean'] = df['actual_score'] > mean_score
        df['predicted_above_mean'] = df['predicted_score'] > mean_score
        direction_accuracy = (df['actual_above_mean'] == df['predicted_above_mean']).sum() / len(df) * 100
        
        # Over/under prediction
        over_predictions = len(df[df['error'] > 0]) / len(df) * 100
        under_predictions = len(df[df['error'] < 0]) / len(df) * 100
        
        metrics = {
            'total_anime': len(df),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'within_0_5': within_0_5,
            'within_1_0': within_1_0,
            'within_1_5': within_1_5,
            'within_ci': within_ci,
            'direction_accuracy': direction_accuracy,
            'over_predictions': over_predictions,
            'under_predictions': under_predictions,
        }
        
        # Print metrics
        print(f"📊 Total Anime Tested: {metrics['total_anime']}")
        print(f"\n🎯 Error Metrics:")
        print(f"   Mean Absolute Error (MAE): {metrics['mae']:.3f}")
        print(f"   Root Mean Squared Error (RMSE): {metrics['rmse']:.3f}")
        print(f"   Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"   R² Score: {metrics['r2']:.3f}")
        
        print(f"\n✅ Accuracy Bands:")
        print(f"   Within ±0.5 points: {metrics['within_0_5']:.1f}%")
        print(f"   Within ±1.0 points: {metrics['within_1_0']:.1f}%")
        print(f"   Within ±1.5 points: {metrics['within_1_5']:.1f}%")
        print(f"   Within Confidence Interval: {metrics['within_ci']:.1f}%")
        
        print(f"\n📈 Prediction Bias:")
        print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"   Over-predictions: {metrics['over_predictions']:.1f}%")
        print(f"   Under-predictions: {metrics['under_predictions']:.1f}%")
        
        print("=" * 70)
        
        self.metrics = metrics
        return metrics
    
    def analyze_by_category(self):
        """
        Analyze accuracy by different categories
        """
        print("\n📊 Category Analysis...")
        print("=" * 70)
        
        df = self.results_df
        
        # Analyze by type
        print("\n🎬 Accuracy by Anime Type:")
        type_analysis = df.groupby('type').agg({
            'absolute_error': ['mean', 'count']
        }).round(3)
        for anime_type, row in type_analysis.iterrows():
            if row[('absolute_error', 'count')] >= 5:  # Only types with 5+ anime
                print(f"   {anime_type}: MAE = {row[('absolute_error', 'mean')]:.3f} (n={int(row[('absolute_error', 'count')])})")
        
        # Analyze by score range
        print("\n⭐ Accuracy by Score Range:")
        df['score_range'] = pd.cut(df['actual_score'], 
                                    bins=[0, 6, 7, 8, 10], 
                                    labels=['Low (0-6)', 'Medium (6-7)', 'Good (7-8)', 'Excellent (8-10)'])
        score_analysis = df.groupby('score_range').agg({
            'absolute_error': ['mean', 'count']
        }).round(3)
        for score_range, row in score_analysis.iterrows():
            print(f"   {score_range}: MAE = {row[('absolute_error', 'mean')]:.3f} (n={int(row[('absolute_error', 'count')])})")
        
        # Best and worst predictions
        print("\n🏆 Best Predictions (smallest error):")
        best = df.nsmallest(5, 'absolute_error')
        for _, row in best.iterrows():
            print(f"   {row['title']}: Actual={row['actual_score']:.2f}, Predicted={row['predicted_score']:.2f}, Error={row['absolute_error']:.3f}")
        
        print("\n😬 Worst Predictions (largest error):")
        worst = df.nlargest(5, 'absolute_error')
        for _, row in worst.iterrows():
            print(f"   {row['title']}: Actual={row['actual_score']:.2f}, Predicted={row['predicted_score']:.2f}, Error={row['absolute_error']:.3f}")
        
        print("=" * 70)
    
    def create_visualizations(self):
        """
        Create comprehensive visualization plots
        """
        print("\n📊 Creating Visualizations...")
        
        df = self.results_df
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Model Validation Results - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter
        axes[0,0].scatter(df['actual_score'], df['predicted_score'], alpha=0.5)
        axes[0,0].plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
        axes[0,0].set_xlabel('Actual Score')
        axes[0,0].set_ylabel('Predicted Score')
        axes[0,0].set_title('Actual vs Predicted Scores')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Error Distribution
        axes[0,1].hist(df['error'], bins=50, edgecolor='black', alpha=0.7)
        axes[0,1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
        axes[0,1].set_xlabel('Prediction Error')
        axes[0,1].set_ylabel('Count')
        axes[0,1].set_title('Error Distribution')
        axes[0,1].legend()
        
        # 3. Absolute Error Distribution
        axes[0,2].hist(df['absolute_error'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0,2].axvline(df['absolute_error'].mean(), color='red', linestyle='--', lw=2, label=f'Mean: {df["absolute_error"].mean():.3f}')
        axes[0,2].set_xlabel('Absolute Error')
        axes[0,2].set_ylabel('Count')
        axes[0,2].set_title('Absolute Error Distribution')
        axes[0,2].legend()
        
        # 4. Residual Plot
        axes[1,0].scatter(df['predicted_score'], df['error'], alpha=0.5)
        axes[1,0].axhline(0, color='red', linestyle='--', lw=2)
        axes[1,0].set_xlabel('Predicted Score')
        axes[1,0].set_ylabel('Residual (Error)')
        axes[1,0].set_title('Residual Plot')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Error by Actual Score
        axes[1,1].scatter(df['actual_score'], df['absolute_error'], alpha=0.5)
        axes[1,1].set_xlabel('Actual Score')
        axes[1,1].set_ylabel('Absolute Error')
        axes[1,1].set_title('Error by Actual Score')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Confidence Interval Coverage
        ci_data = [
            self.metrics['within_0_5'],
            self.metrics['within_1_0'],
            self.metrics['within_1_5'],
            self.metrics['within_ci']
        ]
        labels = ['±0.5', '±1.0', '±1.5', '95% CI']
        axes[1,2].bar(labels, ci_data, color=['green', 'blue', 'orange', 'purple'], alpha=0.7)
        axes[1,2].set_ylabel('Percentage (%)')
        axes[1,2].set_title('Accuracy Within Error Bands')
        axes[1,2].set_ylim([0, 100])
        for i, v in enumerate(ci_data):
            axes[1,2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 7. Error by Type
        type_errors = df.groupby('type')['absolute_error'].mean().sort_values()
        axes[2,0].barh(range(len(type_errors)), type_errors.values, color='skyblue', alpha=0.7)
        axes[2,0].set_yticks(range(len(type_errors)))
        axes[2,0].set_yticklabels(type_errors.index)
        axes[2,0].set_xlabel('Mean Absolute Error')
        axes[2,0].set_title('Error by Anime Type')
        
        # 8. Score Range Analysis
        score_range_errors = df.groupby('score_range')['absolute_error'].mean()
        axes[2,1].bar(range(len(score_range_errors)), score_range_errors.values, 
                     color=['red', 'orange', 'lightgreen', 'green'], alpha=0.7)
        axes[2,1].set_xticks(range(len(score_range_errors)))
        axes[2,1].set_xticklabels(score_range_errors.index, rotation=45, ha='right')
        axes[2,1].set_ylabel('Mean Absolute Error')
        axes[2,1].set_title('Error by Score Range')
        
        # 9. Cumulative Error Distribution
        sorted_errors = np.sort(df['absolute_error'])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[2,2].plot(sorted_errors, cumulative, linewidth=2)
        axes[2,2].set_xlabel('Absolute Error')
        axes[2,2].set_ylabel('Cumulative Percentage (%)')
        axes[2,2].set_title('Cumulative Error Distribution')
        axes[2,2].grid(True, alpha=0.3)
        axes[2,2].axvline(1.0, color='red', linestyle='--', alpha=0.5, label='1.0 error')
        axes[2,2].legend()
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'model_validation_results.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_file}")
        plt.show()
    
    def save_results(self):
        """
        Save results to CSV file
        """
        print("\n💾 Saving Results...")
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_validation_results_{timestamp}.csv"
        
        # Sort by absolute error (worst predictions first)
        self.results_df_sorted = self.results_df.sort_values('absolute_error', ascending=False)
        
        # Save to CSV
        self.results_df_sorted.to_csv(output_file, index=False)
        
        print(f"✅ Results saved: {output_file}")
        print(f"📊 Columns: {', '.join(self.results_df.columns.tolist())}")
        
        return output_file
    
    def generate_report(self):
        """
        Generate a comprehensive text report
        """
        print("\n📄 Generating Comprehensive Report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_validation_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ANIME SCORE PREDICTION MODEL - VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.predictor.model_name}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Total Anime Tested: {self.metrics['total_anime']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("ACCURACY METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Error Metrics:\n")
            f.write(f"  - Mean Absolute Error (MAE): {self.metrics['mae']:.3f}\n")
            f.write(f"  - Root Mean Squared Error (RMSE): {self.metrics['rmse']:.3f}\n")
            f.write(f"  - Mean Absolute Percentage Error (MAPE): {self.metrics['mape']:.2f}%\n")
            f.write(f"  - R² Score: {self.metrics['r2']:.3f}\n\n")
            
            f.write("Accuracy Bands:\n")
            f.write(f"  - Within ±0.5 points: {self.metrics['within_0_5']:.1f}%\n")
            f.write(f"  - Within ±1.0 points: {self.metrics['within_1_0']:.1f}%\n")
            f.write(f"  - Within ±1.5 points: {self.metrics['within_1_5']:.1f}%\n")
            f.write(f"  - Within 95% Confidence Interval: {self.metrics['within_ci']:.1f}%\n\n")
            
            f.write("Prediction Bias:\n")
            f.write(f"  - Direction Accuracy: {self.metrics['direction_accuracy']:.1f}%\n")
            f.write(f"  - Over-predictions (predicted > actual): {self.metrics['over_predictions']:.1f}%\n")
            f.write(f"  - Under-predictions (predicted < actual): {self.metrics['under_predictions']:.1f}%\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Interpretation based on metrics
            if self.metrics['within_1_0'] >= 80:
                f.write("✅ EXCELLENT: Model predictions are highly accurate (80%+ within 1 point)\n\n")
            elif self.metrics['within_1_0'] >= 70:
                f.write("👍 GOOD: Model predictions are reliable (70%+ within 1 point)\n\n")
            elif self.metrics['within_1_0'] >= 60:
                f.write("😊 FAIR: Model predictions are decent (60%+ within 1 point)\n\n")
            else:
                f.write("⚠️  NEEDS IMPROVEMENT: Model accuracy below 60% (within 1 point)\n\n")
            
            if abs(self.metrics['over_predictions'] - 50) < 10:
                f.write("✅ Model is well-balanced (no significant over/under prediction bias)\n\n")
            elif self.metrics['over_predictions'] > 60:
                f.write("⚠️  Model tends to over-predict scores\n\n")
            else:
                f.write("⚠️  Model tends to under-predict scores\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("For Portfolio/Resume:\n")
            f.write(f"  - 'Achieved {self.metrics['within_1_0']:.0f}% prediction accuracy (±1.0 point)'\n")
            f.write(f"  - 'MAE of {self.metrics['mae']:.2f} on 10-point scale'\n")
            f.write(f"  - 'R² score of {self.metrics['r2']:.2f}'\n\n")
            
            f.write("Model Strengths:\n")
            # Analyze which categories perform best
            type_errors = self.results_df.groupby('type')['absolute_error'].mean().sort_values()
            best_type = type_errors.index[0]
            f.write(f"  - Best performance on {best_type} anime\n")
            f.write(f"  - {self.metrics['within_ci']:.0f}% of predictions within confidence interval\n\n")
            
            f.write("=" * 80 + "\n")
            
        print(f"✅ Report saved: {report_file}")
        
        return report_file
    
    def run_full_validation(self):
        """
        Run the complete validation pipeline
        """
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL MODEL VALIDATION")
        print("=" * 70)
        
        # Test all anime
        results_df = self.test_all_anime()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Category analysis
        self.analyze_by_category()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        csv_file = self.save_results()
        
        # Generate report
        report_file = self.generate_report()
        
        print("\n" + "=" * 70)
        print("✅ VALIDATION COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Total Anime: {metrics['total_anime']}")
        print(f"   MAE: {metrics['mae']:.3f}")
        print(f"   Accuracy (±1.0): {metrics['within_1_0']:.1f}%")
        print(f"\n📁 Files Generated:")
        print(f"   - {csv_file}")
        print(f"   - {report_file}")
        print(f"   - model_validation_results.png")
        print("\n🎉 Use these results for your portfolio and README!")
        print("=" * 70 + "\n")


def main():
    """
    Main execution function
    """
    validator = ModelValidator()
    validator.run_full_validation()


if __name__ == "__main__":
    main()