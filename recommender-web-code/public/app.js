// Movie Recommender frontend — ports app.py's Streamlit UI to plain HTML/JS.
// Client-side search over movies.json; recommendations come from POST /api/recommend.

const EMOJIS = {
  Action: "💥", Comedy: "😂", Drama: "🎭", Horror: "👻",
  Romance: "❤️", "Sci-Fi": "🚀", Animation: "🎨",
  Adventure: "🗺️", Thriller: "🔪", Crime: "🕵️",
  Fantasy: "🧙", Children: "👦", Documentary: "🎥",
  Musical: "🎵", Western: "🤠", War: "⚔️", Mystery: "🔍",
};

const MAX_RESULTS = 50;
let MOVIES = [];
let MOVIES_BY_TITLE = new Map();
const selected = [null, null, null]; // movie objects, one per picker slot

async function init() {
  const res = await fetch("movies.json");
  MOVIES = await res.json();
  for (const m of MOVIES) {
    const key = m.title.toLowerCase();
    if (!MOVIES_BY_TITLE.has(key)) MOVIES_BY_TITLE.set(key, m);
  }

  const row = document.getElementById("picker-row");
  for (let i = 0; i < 3; i++) {
    row.appendChild(buildPicker(i));
  }
  renderSuggested(); // initial random suggestions
}

function buildPicker(index) {
  const wrap = document.createElement("div");
  wrap.className = "picker";
  wrap.innerHTML = `
    <label>Movie ${index + 1}</label>
    <input type="text" placeholder="Type part of a movie title..." autocomplete="off" />
    <div class="picker-caption" style="display:none;"></div>
    <div class="picker-results"></div>
  `;
  const input = wrap.querySelector("input");
  const resultsBox = wrap.querySelector(".picker-results");
  const caption = wrap.querySelector(".picker-caption");

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    resultsBox.innerHTML = "";
    if (!q) {
      resultsBox.classList.remove("open");
      return;
    }
    const matches = MOVIES.filter((m) => m.title.toLowerCase().includes(q)).slice(0, MAX_RESULTS);
    if (matches.length === 0) {
      resultsBox.classList.remove("open");
      return;
    }
    for (const m of matches) {
      const item = document.createElement("div");
      item.className = "picker-result-item";
      item.textContent = m.year ? `${m.title} (${m.year})` : m.title;
      item.addEventListener("click", () => {
        selected[index] = m;
        input.value = item.textContent;
        resultsBox.classList.remove("open");
        caption.style.display = m.hasFactors ? "none" : "block";
        caption.textContent = "Metadata-only movie: it can be displayed, but it will not affect SVD recommendations.";
        onSelectionChanged();
      });
      resultsBox.appendChild(item);
    }
    resultsBox.classList.add("open");
  });

  input.addEventListener("blur", () => {
    setTimeout(() => resultsBox.classList.remove("open"), 150);
  });

  return wrap;
}

function movieCardHtml(m, badge) {
  const genres = m.genres || [];
  const icon = genres.length ? (EMOJIS[genres[0]] || "🎬") : "🎬";
  const tags = genres.slice(0, 3).map((g) => `<span class="genre-tag">${escapeHtml(g)}</span>`).join("");
  const badgeHtml = badge ? `<div class="badge">${escapeHtml(badge)}</div>` : "";
  const year = m.year != null ? m.year : "—";
  return `
    <div class="movie-card">
      ${badgeHtml}
      <div class="movie-emoji">${icon}</div>
      <div class="movie-title">${escapeHtml(m.title)}</div>
      <div class="movie-year">${year}</div>
      <div>${tags}</div>
    </div>
  `;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function onSelectionChanged() {
  const chosen = selected.filter(Boolean);

  // Pills
  const pillsEl = document.getElementById("pills");
  pillsEl.innerHTML = chosen
    .map((m) => `<span class="selected-pill"><span class="pill-dot"></span>${escapeHtml(m.title)}</span>`)
    .join("");

  // Preview cards
  const cardsEl = document.getElementById("selected-cards");
  cardsEl.innerHTML = chosen.map((m, i) => movieCardHtml(m, `Pick #${i + 1}`)).join("");

  if (chosen.length === 3) {
    fetchRecommendations(chosen.map((m) => m.title));
  } else {
    renderSuggested();
  }
}

async function fetchRecommendations(titles) {
  const note = document.getElementById("suggested-note");
  const cardsEl = document.getElementById("suggested-cards");
  note.textContent = "Loading suggestions...";
  cardsEl.innerHTML = "";
  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ titles }),
    });
    const data = await res.json();
    const results = data.results || [];
    if (results.length === 0) {
      note.textContent = "No suggestions found for that combination — try different movies.";
      return;
    }
    note.textContent = "";
    cardsEl.innerHTML = results
      .map((r) => movieCardHtml({ title: r.title, year: r.year, genres: (r.genres || "").split(", ").filter(Boolean) }))
      .join("");
  } catch (e) {
    note.textContent = "Couldn't load suggestions right now — please try again.";
  }
}

function renderSuggested() {
  const note = document.getElementById("suggested-note");
  const cardsEl = document.getElementById("suggested-cards");
  const chosenIds = new Set(selected.filter(Boolean).map((m) => m.id));
  note.textContent = "Pick 3 movies above to get personalized suggestions. Meanwhile, here are a few to explore:";

  // Simple seeded-ish random sample from the pool of movies not already picked.
  const pool = MOVIES.filter((m) => !chosenIds.has(m.id));
  const sample = [];
  const used = new Set();
  while (sample.length < 3 && sample.length < pool.length) {
    const idx = Math.floor(Math.random() * pool.length);
    if (used.has(idx)) continue;
    used.add(idx);
    sample.push(pool[idx]);
  }
  cardsEl.innerHTML = sample.map((m) => movieCardHtml(m)).join("");
}

init();
