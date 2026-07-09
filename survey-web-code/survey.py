import os
import datetime

import requests
import pandas as pd
import streamlit as st

# Where survey responses are sent. Paste your deployed Google Apps Script web
# app URL here, or set it in .streamlit/secrets.toml as APPS_SCRIPT_URL, or as
# an environment variable. See apps_script.gs for the script to deploy.
APPS_SCRIPT_URL = (
    st.secrets.get("APPS_SCRIPT_URL", "")
    if hasattr(st, "secrets")
    else ""
) or os.environ.get("APPS_SCRIPT_URL", "")


st.set_page_config(page_title="Movie Survey", page_icon="🎬", layout="centered")


@st.cache_data
def load_movies():
    # Only the columns the survey needs, so loading the large dataset stays fast.
    cols = ["MovieID", "Title", "Year", "Genre1", "Genre2", "Genre3"]
    # dataset.csv lives in the project root, one level above this folder.
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, "..", "dataset.csv"), usecols=cols)
    df = df.dropna(subset=["Title"]).reset_index(drop=True)
    df["display_title"] = df.apply(
        lambda r: f"{r['Title']} ({int(r['Year'])})" if pd.notna(r["Year"]) else str(r["Title"]),
        axis=1,
    )
    return df


def movie_picker(label, key, exclude):
    """Search dataset.csv by title, then pick one movie. Returns the row or None."""
    query = st.text_input(f"חיפוש – {label}", key=f"{key}_q", placeholder="הקלידו חלק משם הסרט...")
    if not query.strip():
        return None
    mask = movies["Title"].astype(str).str.contains(query.strip(), case=False, regex=False, na=False)
    options = movies.loc[mask & ~movies["MovieID"].isin(exclude), "MovieID"].head(100).astype(int).tolist()
    if not options:
        st.caption("לא נמצאו סרטים מתאימים.")
        return None
    selected = st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda mid: movies.loc[movies["MovieID"] == mid, "display_title"].iloc[0],
    )
    return movies[movies["MovieID"] == int(selected)].iloc[0]


movies = load_movies()

st.title("🎬 סקר סרטים")
st.write("בחרו שלושה סרטים שאתם הכי אוהבים, ודרגו אותם מ-1 (הכי אהוב) עד 3.")

name = st.text_input("השם שלך", key="name", placeholder="איך קוראים לך?")

st.subheader("מקום 1 🥇")
row1 = movie_picker("הסרט המדורג #1", "p1", exclude=[])

st.subheader("מקום 2 🥈")
exclude2 = [int(row1["MovieID"])] if row1 is not None else []
row2 = movie_picker("הסרט המדורג #2", "p2", exclude=exclude2)

st.subheader("מקום 3 🥉")
exclude3 = exclude2 + ([int(row2["MovieID"])] if row2 is not None else [])
row3 = movie_picker("הסרט המדורג #3", "p3", exclude=exclude3)

rows = [row1, row2, row3]
all_chosen = all(r is not None for r in rows) and bool(name.strip())

if all_chosen:
    st.markdown("#### הבחירות שלכם:")
    for i, r in enumerate(rows, start=1):
        st.markdown(f"**#{i}** — {r['display_title']}")

if st.button("שליחת הסקר", type="primary", disabled=not all_chosen):
    if not APPS_SCRIPT_URL:
        st.error("חסרה כתובת Apps Script. הגדירו APPS_SCRIPT_URL ב-secrets או כמשתנה סביבה.")
    else:
        payload = {"timestamp": datetime.datetime.now().isoformat(), "name": name.strip()}
        for i, r in enumerate(rows, start=1):
            payload[f"rank{i}_movie_id"] = int(r["MovieID"])
            payload[f"rank{i}_title"] = str(r["Title"])
            payload[f"rank{i}_year"] = "" if pd.isna(r["Year"]) else int(r["Year"])
        try:
            resp = requests.post(APPS_SCRIPT_URL, json=payload, timeout=15)
            resp.raise_for_status()
            st.success("תודה! הסקר נשמר בהצלחה. 🎉")
            st.balloons()
        except requests.RequestException as e:
            st.error(f"השמירה נכשלה: {e}")

if not all_chosen:
    st.info("יש להזין שם ולבחור שלושה סרטים שונים כדי לשלוח.")
