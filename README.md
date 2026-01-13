# 🎌 Anime Score Predictor

A data analysis and machine learning project that predicts MyAnimeList (MAL) scores for anime. This project emphasizes **exploratory data analysis**, **data visualization**, and **statistical insights** to understand what makes anime successful, with machine learning as a predictive tool.

**🚀 [Live Demo](https://anime-score-predictor.streamlit.app/)**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

---
## 📋 Table of Contents

- [Overview](#overview)
- [Data Analysis Highlights](#data-analysis-highlights)
---
## 🎯 Overview

This project analyzes **3000+ anime** from MyAnimeList to uncover patterns in what makes anime successful. Through comprehensive data analysis and visualization, I identified key factors influencing anime ratings and built a predictive model to forecast scores for new anime.

**Project Goals:**
1. Perform in-depth exploratory data analysis on anime characteristics
2. Identify statistical relationships between features and ratings
3. Create data visualization to present findings
4. Build a machine learning model to predict scores of unreleased anime
5. Deploy an accessible, simple web interface for predictions

---

## 📊 Data Analysis Highlights

### Dataset Overview
- **3000+ anime** collected from MyAnimeList via Jikan API
- **150+ features** engineered from raw data
- **Score range:** 2.23 - 9.29 (10-point scale)
- **Time period:** 1960s - 2025
- **Types:** TV, Movies, OVAs, ONAs, Specials

- ### Analysis Process

- #### 1. **Data Collection Strategy**
Implemented a multi-faceted collection approach to ensure dataset quality:

```
📈 Top-rated anime     --→ Capture excellent examples
🌸 Seasonal anime      --→ Time-based patterns
🏷️ Genre-base          --→ Comprehensive genre coverage
📊 Score ranges        --→ Balanced distribution
⭐ Popularity tiers    --→ Different audience sizes
🎬 Type diversity      --→ Format variations
🎲 Random sampling     --→ Unbiased diversity
```

**Key Metric:** Minimum 100 users scored per anime (ensures statistical validity)

#### 2. **Data Cleaning & Preprocessing**
- Missing value analysis and imputation strategies
- Outlier detection using IQR method
- Duplicate removal (by MAL ID)
- Data type standardization
- Text preprocessing for genres/studios

#### 3. **Feature Engineering**
Created meaningful features through domain knowledge:

**Temporal Features:**
- `anime_age`: Years since release
- `is_recent`: Released within last 5 years
- Seasonal patterns (winter, spring, summer, fall)

**Engagement Metrics:**
- `engagement_rate`: scored_by / members ratio
- Popularity tiers (Top, High, Medium, Low)
- Favorites-to-members ratio

**Content Features:**
- 50+ binary genre indicators
- 20 top studio indicators
- Episode categorization (Short, Standard, Long, Very Long)
- Source material encoding (Manga, Light Novel, Original, etc.)

---
