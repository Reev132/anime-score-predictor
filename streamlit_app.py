import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

from scripts.anime_lookup import AnimeAPIClient
from scripts.prediction_interface import AnimeScorePredictionInterface

# Page config
st.set_page_config(
    page_title="Anime Score Predictor",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.25rem;
}
.sub-header {
    text-align: center;
    color: #777;
    margin-bottom: 2rem;
}
.score-display {
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
}
.stButton button {
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "current_search" not in st.session_state:
    st.session_state.current_search = ""

if "selected_anime" not in st.session_state:
    st.session_state.selected_anime = None

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "api_client" not in st.session_state:
    with st.spinner("Initializing API client"):
        st.session_state.api_client = AnimeAPIClient()

if "predictor" not in st.session_state:
    with st.spinner("Loading ML model"):
        try:
            st.session_state.predictor = AnimeScorePredictionInterface()
            st.session_state.model_loaded = True
        except Exception as e:
            st.session_state.model_loaded = False
            st.error(f"Failed to load model: {e}")

# Header
st.markdown('<div class="main-header">Anime Score Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict MyAnimeList scores using machine learning</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "Search for any anime to predict its MAL score. "
        "The model uses features like genre, studio, episode count, and popularity metrics."
    )

    st.divider()
    st.header("Model Info")
    if st.session_state.model_loaded:
        st.success("Model loaded")
        st.write(f"**{st.session_state.predictor.model_name}**")
        st.write(f"RMSE: {st.session_state.predictor.model_results['test_rmse']:.3f}")
        st.write(f"R²: {st.session_state.predictor.model_results['test_r2']:.3f}")
    else:
        st.error("Model not available")

    st.divider()
    if st.button("Random Anime", use_container_width=True):
        anime = st.session_state.api_client.get_random_anime()
        if anime:
            st.session_state.selected_anime = anime
            st.session_state.show_suggestions = False
            st.rerun()

# Search
st.header("Search")

query = st.text_input(
    "Search anime",
    value=st.session_state.current_search,
    placeholder="Type any part of the anime name (e.g., 'oshi no ko s3', 'attack titan')",
    label_visibility="collapsed",
    key="search_input"
)

# Update current search
if query != st.session_state.current_search:
    st.session_state.current_search = query
    st.session_state.show_suggestions = True
    st.session_state.selected_anime = None

# Get suggestions when typing
suggestions = []
if query and len(query) >= 2 and st.session_state.show_suggestions:
    with st.spinner("Searching"):
        suggestions = st.session_state.api_client.get_anime_suggestions(query, limit=10)

# Show suggestions
if suggestions and st.session_state.show_suggestions:
    st.markdown("**Select an anime:**")
    
    for i, s in enumerate(suggestions):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Build display text
            display_parts = [s['title']]
            if s.get('year'):
                display_parts.append(f"({s['year']})")
            if s.get('type'):
                display_parts.append(f"[{s['type']}]")
            
            display_text = " ".join(display_parts)
            
            if st.button(display_text, key=f"suggestion_{i}", use_container_width=True):
                # Fetch full anime data
                with st.spinner("Loading anime data"):
                    anime = st.session_state.api_client.get_anime_by_id(s['mal_id'])
                    if anime:
                        st.session_state.selected_anime = anime
                        st.session_state.show_suggestions = False
                        st.rerun()
        
        with col2:
            if s.get("score"):
                st.caption(f"★ {s['score']:.1f}")
            else:
                st.caption("Not rated")

# Make prediction if anime is selected
if st.session_state.selected_anime and st.session_state.model_loaded:
    anime = st.session_state.selected_anime
    
    # Run prediction
    with st.spinner("Predicting score"):
        result = st.session_state.predictor.predict_score(anime)
    
    # Add to history
    if not st.session_state.prediction_history or st.session_state.prediction_history[0]["anime_data"]["mal_id"] != anime["mal_id"]:
        st.session_state.prediction_history.insert(0, {
            "anime_data": anime,
            "result": result
        })
        st.session_state.prediction_history = st.session_state.prediction_history[:10]

# Show results
if st.session_state.prediction_history:
    st.divider()
    st.header("Prediction Result")

    entry = st.session_state.prediction_history[0]
    anime = entry["anime_data"]
    result = entry["result"]

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        if anime.get("images"):
            st.image(anime["images"], use_container_width=True)

    with c2:
        st.subheader(anime["title"])
        
        # English title if different
        if anime.get("title_english") and anime["title_english"] != anime["title"]:
            st.caption(anime["title_english"])
        
        score = result["predicted_score"]
        # Add color coding to score
        score_color = "#22c55e" if score >= 8 else "#eab308" if score >= 7 else "#ef4444"
        st.markdown(f'<div class="score-display" style="color:{score_color}">{score:.2f} / 10</div>', unsafe_allow_html=True)

        low, high = result["confidence_interval"]
        st.info(f"95% confidence interval: **{low:.2f} – {high:.2f}**")
        
        # Basic info
        info_parts = []
        if anime.get("type"):
            info_parts.append(anime["type"])
        if anime.get("episodes"):
            info_parts.append(f"{anime['episodes']} episodes")
        if anime.get("year"):
            info_parts.append(str(anime["year"]))
        
        if info_parts:
            st.caption(" • ".join(info_parts))

    with c3:
        st.metric("Model", result["model_used"])
        st.metric("RMSE", f"{result['model_rmse']:.3f}")
        
        # Add MAL link button
        if anime.get("url"):
            st.link_button("View on MAL", anime["url"], use_container_width=True)

        # Compare with actual score if available
        actual_score = anime.get("score")
        if actual_score is not None:
            predicted = result["predicted_score"]
            error = abs(predicted - actual_score)

            st.divider()
            st.metric("Actual MAL Score", f"{actual_score:.2f}")
            st.metric("Prediction Error", f"{error:.2f}")

            if error <= 0.5:
                st.success("Very accurate")
            elif error <= 1.0:
                st.info("Reasonably accurate")
            else:
                st.warning("Noticeable deviation")
        else:
            st.caption("No official score yet")

    # Additional details
    with st.expander("Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            if anime.get("genres"):
                st.write("**Genres:**")
                st.write(", ".join(anime["genres"]))
            
            if anime.get("studios"):
                st.write("**Studios:**")
                st.write(", ".join(anime["studios"]))
        
        with col2:
            if anime.get("source"):
                st.write("**Source:**")
                st.write(anime["source"])
            
            if anime.get("status"):
                st.write("**Status:**")
                st.write(anime["status"])
        
        # Add synopsis
        if anime.get("synopsis"):
            st.divider()
            st.write("**Synopsis:**")
            st.write(anime["synopsis"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search another anime", use_container_width=True):
            st.session_state.selected_anime = None
            st.session_state.show_suggestions = True
            st.session_state.current_search = ""
            st.rerun()
    
    with col2:
        if st.button("Clear history", use_container_width=True):
            st.session_state.prediction_history.clear()
            st.session_state.selected_anime = None
            st.session_state.show_suggestions = True
            st.session_state.current_search = ""
            st.rerun()

elif query and len(query) >= 2 and not suggestions and not st.session_state.show_suggestions:
    st.info("No results found. Try a different search term.")

elif not query:
    st.info("Start typing to search for anime.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#777'>Built with Streamlit • Data from MyAnimeList</div>",
    unsafe_allow_html=True
)