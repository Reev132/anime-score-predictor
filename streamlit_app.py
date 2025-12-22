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
    .suggestion-button {
        width: 100%;
        text-align: left;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'current_search' not in st.session_state:
    st.session_state.current_search = ""

if 'selected_anime' not in st.session_state:
    st.session_state.selected_anime = None

if 'trigger_prediction' not in st.session_state:
    st.session_state.trigger_prediction = False

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
    1. Start typing an anime name
    2. Select from suggestions
    3. Get instant ML prediction!
    
    **Perfect for:**
    - Unreleased anime
    - Upcoming seasons
    - Comparing different seasons
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
                st.session_state.selected_anime = random_anime
                st.session_state.current_search = random_anime['title']
                st.session_state.trigger_prediction = True
                st.rerun()

# Main content
st.header("🔍 Search for an Anime")

# Create a form for Enter key support
with st.form(key="search_form", clear_on_submit=False):
    search_query = st.text_input(
        "Start typing to see suggestions...",
        value=st.session_state.current_search,
        placeholder="e.g., Attack on Titan, Death Note, Jujutsu Kaisen...",
        key="search_input",
        label_visibility="collapsed"
    )
    
    # Form submit button (triggered by Enter key)
    form_submit = st.form_submit_button("🚀 Predict Score", type="primary", use_container_width=True)

# Also provide button outside form for manual clicking
predict_button = st.button("🚀 Predict Score (Click)", type="secondary", use_container_width=True, key="manual_predict")

# Combine both triggers
should_predict = form_submit or predict_button or st.session_state.trigger_prediction
st.session_state.trigger_prediction = False  # Reset trigger

# Show live suggestions as user types (case-insensitive)
if search_query and len(search_query) >= 2 and not should_predict:
    with st.spinner("💡 Getting suggestions..."):
        suggestions = st.session_state.api_client.get_anime_suggestions(search_query, limit=8)
        
        if suggestions:
            st.markdown("### 📋 Suggestions:")
            st.caption("Click on an anime to predict its score")
            
            # Create columns for better layout
            for idx, suggestion in enumerate(suggestions):
                # Format suggestion text
                title = suggestion['title']
                year = suggestion.get('year', 'N/A')
                anime_type = suggestion.get('type', 'N/A')
                score = suggestion.get('score')
                score_text = f"⭐ {score:.1f}" if score else "No score"
                
                # Create button for each suggestion
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    suggestion_text = f"**{title}** ({year}) - {anime_type}"
                    if st.button(suggestion_text, key=f"suggestion_{idx}", use_container_width=True):
                        # User clicked a suggestion - fetch full data and predict
                        st.session_state.current_search = title
                        st.session_state.trigger_prediction = True
                        st.rerun()
                
                with col2:
                    st.caption(score_text)
            
            st.divider()

# Handle prediction
if should_predict and search_query and st.session_state.model_loaded:
    # Update current search
    st.session_state.current_search = search_query
    
    with st.spinner(f"🔍 Searching for '{search_query}'..."):
        # Get anime data
        anime_data = st.session_state.api_client.get_full_anime_info_for_prediction(search_query)
        
        if anime_data is None:
            st.error(f"❌ Could not find anime: '{search_query}'")
            st.info("💡 Try:\n- Checking the spelling\n- Using the full title\n- Selecting from suggestions above")
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
        if anime.get('title_japanese') and anime['title_japanese'] != anime['title']:
            st.caption(f"Japanese: {anime['title_japanese']}")
        
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
                actual_score = anime['score']
                st.metric("Actual Score", f"{actual_score:.2f}")
                error = abs(predicted_score - actual_score)
                
                # Calculate if prediction was accurate
                if error <= 0.5:
                    st.success(f"🎯 Very Accurate! Error: {error:.2f}")
                elif error <= 1.0:
                    st.info(f"✅ Accurate! Error: {error:.2f}")
                else:
                    st.warning(f"⚠️ Error: {error:.2f}")
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
                'Actual Score': f"{entry['anime_data'].get('score', 'N/A')}",
                'Type': entry['anime_data'].get('type', 'Unknown'),
                'Year': entry['anime_data'].get('year', 'Unknown')
            }
            for entry in st.session_state.prediction_history[1:]
        ])
        
        st.dataframe(history_df, use_container_width=True, hide_index=True)

else:
    # Welcome message when no predictions yet
    st.info("👆 Start typing an anime name to see suggestions, then press **Enter** or click **Predict Score**!")
    
    st.divider()
    
    # Example predictions
    st.subheader("💡 Try These Examples:")
    
    example_cols = st.columns(3)
    
    examples = [
        "Attack on Titan",
        "Death Note", 
        "Steins Gate",
        "Fullmetal Alchemist Brotherhood",
        "Demon Slayer",
        "Jujutsu Kaisen"
    ]
    
    for idx, example in enumerate(examples):
        with example_cols[idx % 3]:
            if st.button(f"🎬 {example}", key=f"example_{idx}", use_container_width=True):
                st.session_state.current_search = example
                st.session_state.trigger_prediction = True
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🎌 Built with Streamlit | 🤖 Powered by Machine Learning | 📊 Data from MyAnimeList</p>
    <p>💡 <strong>Pro Tips:</strong> Press Enter to predict | Type 2+ characters for suggestions | Works with any case!</p>
    <p>Created as a data science portfolio project</p>
</div>
""", unsafe_allow_html=True)