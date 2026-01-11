import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

from scripts.anime_lookup import AnimeAPIClient
from scripts.prediction_interface import AnimeScorePredictionInterface

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Anime Score Predictor",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
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
.suggestion {
    padding: 0.4rem 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "current_search" not in st.session_state:
    st.session_state.current_search = ""

if "trigger_prediction" not in st.session_state:
    st.session_state.trigger_prediction = False

if "api_client" not in st.session_state:
    with st.spinner("Initializing API client…"):
        st.session_state.api_client = AnimeAPIClient()

if "predictor" not in st.session_state:
    with st.spinner("Loading ML model…"):
        try:
            st.session_state.predictor = AnimeScorePredictionInterface()
            st.session_state.model_loaded = True
        except Exception as e:
            st.session_state.model_loaded = False
            st.error(f"Failed to load model: {e}")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-header">🎌 Anime Score Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict MyAnimeList scores using machine learning</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Type an anime title to predict its MAL score using a trained ML model. "
        "Useful for unreleased anime or comparing seasons."
    )

    st.divider()
    st.header("Model")
    if st.session_state.model_loaded:
        st.success("Model loaded")
        st.write(f"**{st.session_state.predictor.model_name}**")
        st.write(f"RMSE: {st.session_state.predictor.model_results['test_rmse']:.3f}")
        st.write(f"R²: {st.session_state.predictor.model_results['test_r2']:.3f}")
    else:
        st.error("Model not available")

    st.divider()
    if st.button("🎲 Random Anime", use_container_width=True):
        anime = st.session_state.api_client.get_random_anime()
        if anime:
            st.session_state.current_search = anime["title"]
            st.session_state.trigger_prediction = True
            st.rerun()

# --------------------------------------------------
# Search
# --------------------------------------------------
st.header("Search")

with st.form("search_form"):
    query = st.text_input(
        "Anime title",
        value=st.session_state.current_search,
        placeholder="Start typing an anime name…",
        label_visibility="collapsed"
    )
    submit = st.form_submit_button("Predict score", use_container_width=True)

should_predict = submit or st.session_state.trigger_prediction
st.session_state.trigger_prediction = False

# --------------------------------------------------
# Suggestions
# --------------------------------------------------
if query and len(query) >= 2 and not should_predict:
    suggestions = st.session_state.api_client.get_anime_suggestions(query, limit=8)
    if suggestions:
        st.markdown("**Suggestions**")
        for i, s in enumerate(suggestions):
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"{s['title']} ({s.get('year', 'N/A')})",
                    key=f"s_{i}",
                    use_container_width=True
                ):
                    st.session_state.current_search = s["title"]
                    st.session_state.trigger_prediction = True
                    st.rerun()
            with col2:
                if s.get("score"):
                    st.caption(f"⭐ {s['score']:.1f}")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if should_predict and query and st.session_state.model_loaded:
    st.session_state.current_search = query

    with st.spinner("Fetching anime data…"):
        anime = st.session_state.api_client.get_full_anime_info_for_prediction(query)

    if anime is None:
        st.error("Anime not found. Try selecting from suggestions.")
    else:
        with st.spinner("Running prediction…"):
            result = st.session_state.predictor.predict_score(anime)

        st.session_state.prediction_history.insert(0, {
            "anime_data": anime,
            "result": result
        })
        st.session_state.prediction_history = st.session_state.prediction_history[:10]

# --------------------------------------------------
# Results
# --------------------------------------------------
if st.session_state.prediction_history:
    st.divider()
    st.header("Prediction")

    entry = st.session_state.prediction_history[0]
    anime = entry["anime_data"]
    result = entry["result"]

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        if anime.get("images"):
            st.image(anime["images"], use_container_width=True)

    with c2:
        st.subheader(anime["title"])
        score = result["predicted_score"]
        st.markdown(f'<div class="score-display">{score:.2f} / 10</div>', unsafe_allow_html=True)

        low, high = result["confidence_interval"]
        st.info(f"95% confidence interval: **{low:.2f} – {high:.2f}**")

    with c3:
        st.metric("Model", result["model_used"])
        st.metric("RMSE", f"{result['model_rmse']:.3f}")

        # Show comparison only if actual score exists (released anime)
        actual_score = anime.get("score")
        if actual_score is not None:
            predicted = result["predicted_score"]
            error = abs(predicted - actual_score)

            st.divider()
            st.metric("Actual MAL Score", f"{actual_score:.2f}")
            st.metric("Absolute Difference", f"{error:.2f}")

            if error <= 0.5:
                st.success("🎯 Very accurate prediction")
            elif error <= 1.0:
                st.info("✅ Reasonably accurate")
            else:
                st.warning("⚠️ Noticeable deviation")
        else:
            st.caption("⏳ No official MAL score yet")


    st.divider()

    if st.button("Clear results", use_container_width=True):
        st.session_state.prediction_history.clear()
        st.session_state.current_search = ""
        st.rerun()

else:
    st.info("Start typing an anime title to get predictions.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#777'>Built with Streamlit · Data from MyAnimeList</div>",
    unsafe_allow_html=True
)
