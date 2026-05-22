import streamlit as st
import pandas as pd
import random

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

  /* Background & base */
  .stApp { background-color: #faf9f7; color: #1a1a1a; font-family: 'Inter', sans-serif; }

  #MainMenu, footer, header { visibility: hidden; }

  /* Hero */
  .hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #e8e0d5;
    margin-bottom: 2rem;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
  }
  .hero-title span { color: #8b1a1a; }
  .hero-sub {
    color: #888;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
  }

  /* Section headers */
  .section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8e0d5;
    margin-left: 0.75rem;
  }

  /* Movie card */
  .movie-card {
    background: #ffffff;
    border: 1px solid #e8e0d5;
    border-radius: 10px;
    padding: 1.4rem 1.1rem;
    text-align: center;
    transition: transform 0.18s, box-shadow 0.18s;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .movie-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    border-color: #c9a96e;
  }
  .movie-emoji { font-size: 2.4rem; margin-bottom: 0.6rem; }
  .movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
    line-height: 1.35;
  }
  .movie-year {
    font-size: 0.8rem;
    color: #8b1a1a;
    font-weight: 600;
    margin-bottom: 0.5rem;
    letter-spacing: 0.5px;
  }
  .genre-tag {
    display: inline-block;
    background: #f5f0ea;
    border: 1px solid #e8e0d5;
    border-radius: 20px;
    padding: 2px 9px;
    margin: 2px;
    font-size: 0.7rem;
    color: #666;
  }
  .badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #8b1a1a;
    margin-bottom: 6px;
  }

  /* Selected pills */
  .selected-pill {
    display: inline-flex;
    align-items: center;
    background: #fff;
    border: 1px solid #c9a96e;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.83rem;
    color: #1a1a1a;
  }
  .selected-pill .pill-dot {
    width: 7px; height: 7px;
    background: #c9a96e;
    border-radius: 50%;
    margin-right: 7px;
  }

  /* Divider */
  .custom-divider {
    border: none;
    height: 1px;
    background: #e8e0d5;
    margin: 2.2rem 0;
  }

  /* Button */
  .stButton > button {
    background: #8b1a1a;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.4px;
    padding: 0.45rem 1.4rem;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #6e1414; }

  /* Labels */
  label { color: #444 !important; font-size: 0.88rem !important; font-weight: 500 !important; }

  /* Selectbox */
  .stSelectbox [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #ddd !important;
    color: #1a1a1a !important;
    border-radius: 6px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_movies():
    df = pd.read_csv(
        "datasets/movies/movies_clean.csv",
        dtype={"MovieID": int, "Year": "Int64"},
    )
    genre_cols = [c for c in df.columns if c.startswith("Genre")]
    df["genres"] = df[genre_cols].apply(
        lambda row: [g for g in row if pd.notna(g) and g != ""], axis=1
    )
    return df


def get_random_suggestions(df, exclude_ids, n=3):
    """Placeholder — replace with your recommendation function later."""
    pool = df[~df["MovieID"].isin(exclude_ids)]
    return pool.sample(n=min(n, len(pool))).reset_index(drop=True)


def movie_card_html(row, badge=None):
    emojis = {"Action": "💥", "Comedy": "😂", "Drama": "🎭", "Horror": "👻",
              "Romance": "❤️", "Sci-Fi": "🚀", "Animation": "🎨",
              "Adventure": "🗺️", "Thriller": "🔪", "Crime": "🕵️",
              "Fantasy": "🧙", "Children": "👦", "Documentary": "🎥",
              "Musical": "🎵", "Western": "🤠", "War": "⚔️", "Mystery": "🔍"}
    genres = row["genres"]
    icon = emojis.get(genres[0], "🎬") if genres else "🎬"
    genre_tags = "".join(f'<span class="genre-tag">{g}</span>' for g in genres[:3])
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    year = str(row["Year"]) if pd.notna(row["Year"]) else "—"
    return f"""
    <div class="movie-card">
      {badge_html}
      <div class="movie-emoji">{icon}</div>
      <div class="movie-title">{row['Title']}</div>
      <div class="movie-year">{year}</div>
      <div>{genre_tags}</div>
    </div>
    """


# ── App ───────────────────────────────────────────────────────────────────────
movies_df = load_movies()
movie_titles = sorted(movies_df["Title"].tolist())

st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">🎬 Movie <span>Recommender</span></div>
  <div class="hero-sub">Tell us what you love and we'll find your next watch</div>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Pick your movies ───────────────────────────────────────────────
st.markdown('<div class="section-header">Pick 3 Movies You Love</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    pick1 = st.selectbox("Movie 1", ["— Select a movie —"] + movie_titles, key="p1")
with col2:
    pick2 = st.selectbox("Movie 2", ["— Select a movie —"] + movie_titles, key="p2")
with col3:
    pick3 = st.selectbox("Movie 3", ["— Select a movie —"] + movie_titles, key="p3")

selections = [p for p in [pick1, pick2, pick3] if p != "— Select a movie —"]

# Show selected pills
if selections:
    pills_html = "".join(
        f'<span class="selected-pill"><span class="pill-dot"></span>{s}</span>'
        for s in selections
    )
    st.markdown(f'<div style="margin-top:0.8rem;">{pills_html}</div>', unsafe_allow_html=True)

# Preview cards for selected movies
if selections:
    st.markdown("<br>", unsafe_allow_html=True)
    card_cols = st.columns(3)
    for i, title in enumerate(selections):
        row = movies_df[movies_df["Title"] == title].iloc[0]
        with card_cols[i]:
            st.markdown(movie_card_html(row, badge=f"Pick #{i+1}"), unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Section 2: Suggested for You ─────────────────────────────────────────────
st.markdown('<div class="section-header">Suggested for You</div>', unsafe_allow_html=True)

selected_ids = movies_df[movies_df["Title"].isin(selections)]["MovieID"].tolist()

# Re-generate suggestions whenever the selection changes
if st.session_state.get("last_selections") != selections:
    st.session_state.last_selections = selections
    st.session_state.suggestions = get_random_suggestions(movies_df, selected_ids)

sugg_cols = st.columns(3)
for i, (_, row) in enumerate(st.session_state.suggestions.iterrows()):
    with sugg_cols[i]:
        st.markdown(movie_card_html(row), unsafe_allow_html=True)

# Footer note
st.markdown("""
<div style="text-align:center;color:#bbb;font-size:0.75rem;margin-top:3rem;border-top:1px solid #e8e0d5;padding-top:1.5rem;">
  Suggestions are random placeholders — your recommendation engine goes here.
</div>
""", unsafe_allow_html=True)
