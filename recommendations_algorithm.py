import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

RATINGS_PATH = "datasets/movies/raw/ratings.csv"
MOVIES_PATH  = "datasets/movies/movies_clean.csv"
SUBS_DIR     = "subs"
SVD_K        = 50

# Signal weights (must sum to 1.0)
W_CF        = 0.60   # Collaborative filtering
W_DIALOGUE  = 0.25   # Dialogue quality
W_FRANCHISE = 0.15   # Franchise awareness

#dialogue features
def parse_srt(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'[\[\(][^\]\)]*[\]\)]', '', content)
    return re.findall(r"[a-zA-Z']+", content.lower())

def dialogue_quality_score(words):
    if len(words) < 50:
        return None
    total  = len(words)
    unique = len(set(words))
    ttr    = unique / total
    top_freq = Counter(words).most_common(1)[0][1] / total
    return ttr - top_freq  # higher = richer vocab, lower repetition

def load_dialogue_features(subs_dir):
    print("loading dialogue features...")
    raw = {}
    for fname in os.listdir(subs_dir):
        if not fname.endswith('.srt'):
            continue
        mid   = int(fname.split('_')[0])
        words = parse_srt(os.path.join(subs_dir, fname))
        score = dialogue_quality_score(words)
        if score is not None:
            raw[mid] = score

    # Normalize to [0, 1]
    values = np.array(list(raw.values()))
    mn, mx = values.min(), values.max()
    return {mid: (score - mn) / (mx - mn) for mid, score in raw.items()}


#  SVD collaborative filtering
def load_svd_model(ratings_path):
    print("Building SVD model from ratings...")
    if ratings_path.endswith('.csv'):
        df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        df.columns = ['UserID', 'MovieID', 'Rating']
    else:
        ratings = []
        with open(ratings_path) as f:
            for line in f:
                uid, mid, rating, _ = line.strip().split('::')
                ratings.append((int(uid), int(mid), float(rating)))
        df = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating'])
    movie_ids = sorted(df.MovieID.unique())
    user_ids  = sorted(df.UserID.unique())
    user_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = [user_idx[u] for u in df.UserID]
    cols = [movie_idx[m] for m in df.MovieID]
    R    = csr_matrix((df.Rating.values, (rows, cols)),
                      shape=(len(user_ids), len(movie_ids)), dtype=np.float32)

    U, sigma, Vt = svds(R, k=SVD_K)
    movie_factors = (np.diag(sigma) @ Vt).T  # (N_movies, k)
    return movie_ids, movie_factors


# franchise detection


def detect_franchises(movies_df):
    def base_title(title):
        t = re.sub(r'\s+(II|III|IV|V|VI|VII|VIII|IX|X|\d+)$', '', title.strip(), flags=re.IGNORECASE)
        t = re.sub(r':\s.*$', '', t)
        return t.strip().lower()

    movies_df = movies_df.copy()
    movies_df['base'] = movies_df['Title'].apply(base_title)
    groups = movies_df.groupby('base')['MovieID'].apply(list)
    result = {}
    for base, mids in groups.items():
        if len(mids) > 1:
            for mid in mids:
                result[mid] = base
    return result


# recommend


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def recommend(input_titles, movies, movie_ids, movie_factors, dq_map, franchise_map, n=3):
    mid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # find input movie IDs by title (case-insensitive)
    input_ids = []
    for title in input_titles:
        match = movies[movies['Title'].str.lower() == title.lower()]
        if match.empty:
            # partial match fallback
            match = movies[movies['Title'].str.lower().str.contains(title.lower())]
        if match.empty:
            print(f" Could not find movie: '{title}'")
            continue
        mid = int(match.iloc[0].MovieID)
        if mid not in mid_to_idx:
            print(f"  '{match.iloc[0].Title}' has no rating data, skipping.")
            continue
        input_ids.append(mid)
        print(f"   Found: {match.iloc[0].Title} ({int(match.iloc[0].Year)})")

    if len(input_ids) < 1:
        print("No valid input movies found.")
        return []

    # user profile = mean of input movie latent factors
    user_profile = np.mean([movie_factors[mid_to_idx[mid]] for mid in input_ids], axis=0)

    # user's average dialogue quality preference
    user_dq = np.mean([dq_map.get(mid, 0.5) for mid in input_ids])

    # score all candidate movies
    candidates = []
    for i, mid in enumerate(movie_ids):
        if mid in input_ids:
            continue
        row = movies[movies.MovieID == mid]
        if row.empty:
            continue

        # 1. collaborative filtering score
        cf_score = max(0.0, cosine_sim(user_profile, movie_factors[i]))

        # 2. dialogue quality similarity
        movie_dq = dq_map.get(mid, 0.5)
        dq_score = 1.0 - abs(movie_dq - user_dq)

        # 3. franchise awareness
        franchise_score = 0.5  # neutral default
        movie_franchise = franchise_map.get(mid)
        if movie_franchise:
            for input_id in input_ids:
                if franchise_map.get(input_id) == movie_franchise:
                    input_dq = dq_map.get(input_id, 0.5)
                    franchise_score = 1.0 if movie_dq >= input_dq - 0.1 else 0.0
                    break

        score = W_CF * cf_score + W_DIALOGUE * dq_score + W_FRANCHISE * franchise_score
        candidates.append((mid, score, cf_score, dq_score, franchise_score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # build result list
    results = []
    for mid, score, cf, dq, fs in candidates[:n]:
        row    = movies[movies.MovieID == mid].iloc[0]
        genres = [row[f'Genre{i}'] for i in range(1, 9) if pd.notna(row.get(f'Genre{i}'))]
        franchise_note = ""
        if fs == 1.0:
            franchise_note = " [franchise quality maintained ]"
        elif fs == 0.0:
            franchise_note = " [franchise quality declined ]"
        results.append({
            'title':    row['Title'],
            'year':     int(row['Year']),
            'genres':   ', '.join(genres),
            'score':    round(score, 3),
            'cf':       round(cf, 3),
            'dq':       round(dq, 3),
            'franchise': round(fs, 3),
            'note':     franchise_note,
        })
    return results


def main():
    # --- Load all data ---
    movies        = pd.read_csv(MOVIES_PATH)
    dq_map        = load_dialogue_features(SUBS_DIR)
    movie_ids, movie_factors = load_svd_model(RATINGS_PATH)
    franchise_map = detect_franchises(movies)



    while True:
        print("\nEnter 3 movies you like (or 'quit' to exit):")
        titles = []
        for i in range(1, 4):
            t = input(f"  Movie {i}: ").strip()
            if t.lower() == 'quit':
                return
            titles.append(t)


        results = recommend(titles, movies, movie_ids, movie_factors, dq_map, franchise_map)

        if results:
            print("\n recommendations ")
            for rank, r in enumerate(results, 1):
                print(f"\n  {rank}. {r['title']} ({r['year']}){r['note']}")
                print(f"     Genres:  {r['genres']}")
                print(f"     Score:   {r['score']}  "
                      f"(CF: {r['cf']} | Dialogue: {r['dq']} | Franchise: {r['franchise']})")
                print("\n recommendations ")


        again = input("\nTry again? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == '__main__':
    main()
