import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd

# Add scripts directory to path so we can import our modules
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

from scripts.anime_lookup import AnimeAPIClient
from scripts.prediction_interface import AnimeScorePredictionInterface

# Page configuration
st.set_page_config(
    page_title="🎌 Anime Score Predictor",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'current_search' not in st.session_state:
    st.session_state.current_search = ""

if 'api_client' not in st.session_state:
    with st.spinner("🔄 Initializing API client..."):
        st.session_state.api_client = AnimeAPIClient()

if 'predictor' not in st.session_state:
    with st.spinner("🤖 Loading ML model..."):
        try:
            st.session_state.predictor = AnimeScorePredictionInterface()
            st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"❌ Failed to load ML model: {e}")
            st.info("💡 Make sure you've trained the model by running: python scripts/anime_ml_pipeline.py")
            st.session_state.model_loaded = False

# Header
st.markdown('<p class="main-header">🎌 Anime Score Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict MAL scores for any anime using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses machine learning to predict MyAnimeList (MAL) scores for anime.
    
    **How it works:**
    1. Enter an anime name
    2. ML model analyzes features (genres, studios, type, etc.)
    3. Predicts the likely MAL score
    
    **Perfect for:**
    - Unreleased anime
    - Upcoming seasons
    - Comparing potential scores
    """)
    
    st.divider()
    
    st.header("📊 Model Info")
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded")
        st.write(f"**Model:** {st.session_state.predictor.model_name}")
        st.write(f"**Test RMSE:** {st.session_state.predictor.model_results['test_rmse']:.3f}")
        st.write(f"**Test R²:** {st.session_state.predictor.model_results['test_r2']:.3f}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.divider()
    
    st.header("🎲 Try Random")
    if st.button("Get Random Anime", use_container_width=True):
        with st.spinner("🎲 Fetching random anime..."):
            random_anime = st.session_state.api_client.get_random_anime()
            if random_anime:
                st.session_state.current_search = random_anime['title']
                st.rerun()

# Main content
st.header("🔍 Search for an Anime")

# Search input
search_query = st.text_input(
    "Enter anime name:",
    value=st.session_state.current_search,
    placeholder="e.g., Attack on Titan, Death Note, Jujutsu Kaisen...",
    key="search_input"
)

# Predict button
predict_button = st.button("🚀 Predict Score", type="primary", use_container_width=True)

# Handle prediction
if predict_button and search_query and st.session_state.model_loaded:
    # Update current search
    st.session_state.current_search = search_query
    
    with st.spinner(f"🔍 Searching for '{search_query}'..."):
        # Get anime data
        anime_data = st.session_state.api_client.get_full_anime_info_for_prediction(search_query)
        
        if anime_data is None:
            st.error(f"❌ Could not find anime: '{search_query}'")
            st.info("💡 Try checking the spelling or using a different name (English/Japanese)")
        else:
            with st.spinner("🤖 Making prediction..."):
                try:
                    # Make prediction
                    result = st.session_state.predictor.predict_score(anime_data)
                    
                    # Add to history
                    history_entry = {
                        'anime_data': anime_data,
                        'result': result
                    }
                    st.session_state.prediction_history.insert(0, history_entry)
                    
                    # Keep only last 10 predictions
                    st.session_state.prediction_history = st.session_state.prediction_history[:10]
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("💡 This might be due to missing features. Try a different anime.")

# Display results
if st.session_state.prediction_history:
    st.divider()
    st.header("📊 Prediction Results")
    
    # Display most recent prediction
    latest = st.session_state.prediction_history[0]
    anime = latest['anime_data']
    result = latest['result']
    
    # Main prediction display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if anime.get('images'):
            st.image(anime['images'], use_container_width=True)
    
    with col2:
        st.markdown(f"### {anime['title']}")
        if anime.get('title_english') and anime['title_english'] != anime['title']:
            st.caption(f"English: {anime['title_english']}")
        
        # Prediction score
        predicted_score = result['predicted_score']
        st.markdown(f'<p class="score-display">{predicted_score:.2f}/10</p>', unsafe_allow_html=True)
        
        # Confidence interval
        ci_low, ci_high = result['confidence_interval']
        st.info(f"📈 **95% Confidence Interval:** {ci_low:.2f} - {ci_high:.2f}")
        
        # Interpretation
        if predicted_score >= 8.5:
            st.success("✨ **Interpretation:** Excellent anime! Likely to be highly rated.")
        elif predicted_score >= 7.5:
            st.success("👍 **Interpretation:** Very good anime. Should be well-received.")
        elif predicted_score >= 6.5:
            st.info("😊 **Interpretation:** Good anime. Average to above-average reception.")
        elif predicted_score >= 5.5:
            st.warning("😐 **Interpretation:** Mediocre anime. Mixed reception likely.")
        else:
            st.error("😬 **Interpretation:** Below average. May struggle with ratings.")
    
    with col3:
        st.metric("Model Used", result['model_used'])
        st.metric("Model RMSE", f"{result['model_rmse']:.3f}")
        
        if anime.get('is_released'):
            st.success("✅ Released")
            if anime.get('score'):
                st.metric("Actual Score", f"{anime['score']:.2f}")
                error = abs(predicted_score - anime['score'])
                st.metric("Prediction Error", f"{error:.2f}")
        else:
            st.warning("⏳ Unreleased")
    
    # Anime details
    st.divider()
    st.subheader("📺 Anime Details")
    
    detail_cols = st.columns(4)
    
    with detail_cols[0]:
        st.markdown("**Type**")
        st.write(anime.get('type', 'Unknown'))
        st.markdown("**Episodes**")
        st.write(anime.get('episodes', 'Unknown'))
    
    with detail_cols[1]:
        st.markdown("**Year**")
        st.write(anime.get('year', 'Unknown'))
        st.markdown("**Season**")
        st.write(anime.get('season', 'Unknown').title())
    
    with detail_cols[2]:
        st.markdown("**Source**")
        st.write(anime.get('source', 'Unknown'))
        st.markdown("**Rating**")
        st.write(anime.get('rating', 'Unknown'))
    
    with detail_cols[3]:
        st.markdown("**Status**")
        st.write(anime.get('status', 'Unknown'))
        if anime.get('popularity'):
            st.markdown("**Popularity**")
            st.write(f"#{anime['popularity']}")
    
    # Genres and Studios
    col1, col2 = st.columns(2)
    
    with col1:
        if anime.get('genres'):
            st.markdown("**Genres**")
            genre_tags = " ".join([f"`{g}`" for g in anime['genres']])
            st.markdown(genre_tags)
    
    with col2:
        if anime.get('studios'):
            st.markdown("**Studios**")
            studio_tags = " ".join([f"`{s}`" for s in anime['studios']])
            st.markdown(studio_tags)
    
    # Synopsis
    if anime.get('synopsis'):
        st.divider()
        st.subheader("📝 Synopsis")
        st.write(anime['synopsis'])
    
    # MAL Link
    if anime.get('url'):
        st.markdown(f"[🔗 View on MyAnimeList]({anime['url']})")
    
    # Clear results button
    st.divider()
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.current_search = ""
        st.rerun()
    
    # Prediction history
    if len(st.session_state.prediction_history) > 1:
        st.divider()
        st.subheader("📜 Recent Predictions")
        
        history_df = pd.DataFrame([
            {
                'Anime': entry['anime_data']['title'],
                'Predicted Score': f"{entry['result']['predicted_score']:.2f}",
                'Confidence Range': f"{entry['result']['confidence_interval'][0]:.2f} - {entry['result']['confidence_interval'][1]:.2f}",
                'Type': entry['anime_data'].get('type', 'Unknown'),
                'Year': entry['anime_data'].get('year', 'Unknown')
            }
            for entry in st.session_state.prediction_history[1:]
        ])
        
        st.dataframe(history_df, use_container_width=True, hide_index=True)

else:
    # Welcome message when no predictions yet
    st.info("👆 Enter an anime name above and click 'Predict Score' to get started!")
    
    st.divider()
    
    # Example predictions
    st.subheader("💡 Try These Examples:")
    
    example_cols = st.columns(3)
    
    examples = [
        "Attack on Titan",
        "Death Note",
        "Steins Gate",
        "Fullmetal Alchemist Brotherhood",
        "Your Name",
        "Demon Slayer"
    ]
    
    for idx, example in enumerate(examples):
        with example_cols[idx % 3]:
            if st.button(f"🎬 {example}", key=f"example_{idx}", use_container_width=True):
                st.session_state.current_search = example
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎌 Built with Streamlit | 🤖 Powered by Machine Learning | 📊 Data from MyAnimeList</p>
    <p>Created as a data science portfolio project</p>
</div>
""", unsafe_allow_html=True)