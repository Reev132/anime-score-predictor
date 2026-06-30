# 🎌 Anime Score Predictor

A data analysis and machine learning project that predicts MyAnimeList (MAL) scores for anime. The project leans on exploratory data analysis and feature engineering to understand what makes anime score well, with a Random Forest model as the predictive layer and a Streamlit app as the interface.

**[Live Demo](https://anime-score-predictor.streamlit.app/)**

---

## Overview

This project analyzes 3,600+ anime from MyAnimeList to find patterns in what drives high ratings, then uses those patterns to predict scores for both existing and unreleased anime.

**Goals:**

1. Explore anime characteristics and their relationship to score
2. Engineer features that capture genre, studio, format, and audience engagement signals
3. Train and evaluate a regression model on those features
4. Ship a simple web interface where anyone can search an anime and get a prediction

---

## Dataset

- **3,602 anime** collected from MyAnimeList via the Jikan API
- **Score range:** 2.23 – 9.29 (10-point scale)
- **Time period:** 1967 – 2025
- **Types:** TV, Movies, OVAs, ONAs, Specials
- Minimum 100 scoring users per anime, to avoid noisy low-sample entries

Collection was spread across top-rated anime, seasonal pulls, genre coverage, score-range balancing, popularity tiers, format diversity, and random sampling — aimed at avoiding a dataset that's just "popular anime," which would bias the model toward predicting high scores for anything well-known.

---

## Feature Engineering

**Temporal:** `anime_age` (years since release), `is_recent` (released in the last 5 years), season.

**Engagement:** `engagement_rate` (scored_by / members), popularity tier (Top/High/Medium/Low), favorites-to-members ratio.

**Content:** binary indicators for genres and for the 20 most common studios, episode-count category (Short/Standard/Long/Very Long), and source material (Manga, Light Novel, Original, etc.).

---

## Key Insights

**Score distribution** is roughly bell-shaped, centered around 7–8:

```
Excellent (8–10) │ ██░░░░░░░░  694 anime  (19%)
Good (7–8)       │ █████░░░░░  1802 anime (50%)
Medium (6–7)     │ ██░░░░░░░░  777 anime  (22%)
Low (0–6)        │ █░░░░░░░░░  329 anime  (9%)
```

**Top-rated genres** (by average score): Drama (7.49), Mystery (7.49), Adventure (7.36), Supernatural (7.34), Action (7.32) — drama-heavy, narrative-driven genres edge out pure action.

**Popularity metrics matter a lot, but carry a caveat.** `members`, `favorites`, and `scored_by` are some of the strongest predictors in the model — which makes sense, but also means a chunk of the model's accuracy comes from "anime people already engaged with tend to be rated well," not purely from content quality. This is why predictions for **unreleased** anime use a separate mode (see below) that excludes these features entirely, since they don't exist yet for something that hasn't aired.

---

## Model

**Algorithm:** Random Forest, selected after comparing Linear/Ridge/Lasso Regression, Gradient Boosting, and SVR via 5-fold cross-validation.

| Metric | Train | Test  |
| ------ | ----- | ----- |
| RMSE   | 0.194 | 0.501 |
| MAE    | 0.136 | 0.356 |
| R²     | 0.952 | 0.721 |

Test RMSE of ~0.50 means predictions land within about half a point of the actual MAL score on average. The gap between train and test R² (0.95 vs 0.72) indicates some overfitting — performance on new data is solid but noticeably weaker than on data it's already seen, which is worth keeping in mind rather than over-trusting the headline train numbers.

### Pre-release prediction mode

For anime that haven't aired yet, the app automatically excludes `members`, `favorites`, `scored_by`, `popularity`, and `engagement_rate` from the prediction instead of guessing at them. Predictions for unreleased shows are based only on genre, studio, source material, format, episode count, and rating — the things actually knowable before something airs. This is detected automatically from MAL's airing status, no manual toggle needed.

---

## Tech Stack

- **Data & analysis:** pandas, NumPy, matplotlib, seaborn, SciPy
- **Machine learning:** scikit-learn, joblib
- **Web app:** Streamlit
- **Data source:** Jikan API v4 (unofficial MyAnimeList API)

---

## Installation

```bash
git clone https://github.com/Reev132/anime-score-predictor.git
cd anime-score-predictor

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

---

## Project Structure

```
anime-score-predictor/
├── streamlit_app.py              # Web app (main entry point)
├── requirements.txt
├── scripts/
│   ├── anime_lookup.py           # Jikan API client
│   ├── prediction_interface.py   # Model loading + prediction logic
│   ├── analyze_data.py           # EDA and visualization
│   ├── anime_ml_pipeline.py      # Model training
│   ├── validate_model.py         # Model validation suite
│   └── enhanced_collect_data.py  # Data collection
├── models/                       # Trained model (.joblib)
└── data/raw/                     # Collected dataset (.csv)
```

---

## Usage

**Web app:** search for an anime, select it from suggestions, and see the predicted score with a confidence interval — automatically run in pre-release mode if the anime hasn't aired.

**Python:**

```python
from scripts.anime_lookup import AnimeAPIClient
from scripts.prediction_interface import AnimeScorePredictionInterface

client = AnimeAPIClient()
predictor = AnimeScorePredictionInterface()

anime = client.search_anime_by_name("Steins;Gate")
result = predictor.predict_score(anime)

print(f"Predicted: {result['predicted_score']:.2f}")
print(f"Actual: {anime['score']:.2f}")
```

---

## Acknowledgments

- **MyAnimeList** for the dataset this project builds on
- **Jikan API** for free, open access to MAL data
