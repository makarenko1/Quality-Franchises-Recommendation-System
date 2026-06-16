import pandas as pd
import numpy as np

RATINGS_PATH = "datasets/movies-1M/movies_ratings_clean.csv"
MOVIES_PATH  = "datasets/movies-1M/movies_clean.csv"


def load_rating_stats(ratings_path):
    df = pd.read_csv(ratings_path, usecols=['MovieID', 'Rating'])
    stats = df.groupby('MovieID')['Rating'].agg(['count', 'mean'])
    stats.columns = ['vote_count', 'vote_average']
    return stats


def resolve_input_ids(input_titles, movies):
    input_ids = []
    for title in input_titles:
        match = movies[movies['Title'].str.lower() == title.lower()]
        if match.empty:
            match = movies[movies['Title'].str.lower().str.contains(title.lower())]
        if match.empty:
            continue
        input_ids.append(int(match.iloc[0].MovieID))
    return input_ids


def format_results(movies, mids):
    results = []
    for mid in mids:
        row = movies[movies.MovieID == mid]
        if row.empty:
            continue
        row = row.iloc[0]
        genres = [row[f'Genre{i}'] for i in range(1, 9) if pd.notna(row.get(f'Genre{i}'))]
        results.append({
            'title':  row['Title'],
            'year':   int(row['Year']) if pd.notna(row['Year']) else None,
            'genres': ', '.join(genres),
        })
    return results


def recommend_popular(input_titles, movies, stats, n=3):
    # popularity = how many people rated it (vote count)
    input_ids = resolve_input_ids(input_titles, movies)
    ranked = stats.sort_values('vote_count', ascending=False)
    picks = [mid for mid in ranked.index if mid not in input_ids][:n]
    return format_results(movies, picks)


def recommend_highest_rated(input_titles, movies, stats, n=3):
    # IMDB weighted rating (Bayesian average):
    #   WR = (v / (v + m)) * R + (m / (v + m)) * C
    # v = votes for the movie, R = its mean rating,
    # m = minimum votes to be listed, C = mean rating across all movies.
    input_ids = resolve_input_ids(input_titles, movies)
    C = stats['vote_average'].mean()
    m = stats['vote_count'].quantile(0.90)
    qualified = stats[stats['vote_count'] >= m].copy()
    v = qualified['vote_count']
    R = qualified['vote_average']
    qualified['score'] = (v / (v + m)) * R + (m / (v + m)) * C
    ranked = qualified.sort_values('score', ascending=False)
    picks = [mid for mid in ranked.index if mid not in input_ids][:n]
    return format_results(movies, picks)


def recommend_random(input_titles, movies, stats, n=3, seed=None):
    input_ids = resolve_input_ids(input_titles, movies)
    pool = [mid for mid in stats.index if mid not in input_ids]
    rng = np.random.default_rng(seed)
    picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return format_results(movies, [int(mid) for mid in picks])


def recommend_baseline(input_titles, baseline, movies, stats, n=3):
    if baseline == 'popular':
        return recommend_popular(input_titles, movies, stats, n)
    elif baseline == 'highest_rated':
        return recommend_highest_rated(input_titles, movies, stats, n)
    elif baseline == 'random':
        return recommend_random(input_titles, movies, stats, n)
    else:
        raise ValueError(f"unknown baseline: {baseline}")


def main():
    movies = pd.read_csv(MOVIES_PATH, low_memory=False)
    stats  = load_rating_stats(RATINGS_PATH)

    print("Enter 3 movies you like:")
    titles = []
    for i in range(1, 4):
        t = input(f"  Movie {i}: ").strip()
        titles.append(t)

    baseline = input("\nBaseline (popular / highest_rated / random): ").strip()

    print(f"\n--- {baseline} ---")
    for r in recommend_baseline(titles, baseline, movies, stats):
        print(f"  {r['title']} ({r['year']}) - {r['genres']}")


if __name__ == '__main__':
    main()
