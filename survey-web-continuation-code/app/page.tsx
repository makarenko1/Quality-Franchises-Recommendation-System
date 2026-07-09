"use client";

import { FormEvent, useMemo, useState } from "react";
import { surveyData } from "../data/surveyData";

const methods = ["our_system", "popular", "highest_rated", "random"] as const;

type Method = typeof methods[number];

function imdbSearchUrl(movie: string) {
  return "https://www.imdb.com/find/?q=" + encodeURIComponent(movie) + "&s=tt";
}

function seededOrder(seed: string, values: readonly Method[]) {
  return [...values].sort((a, b) => {
    const aKey = `${seed}:${a}`;
    const bKey = `${seed}:${b}`;
    return aKey.localeCompare(bKey);
  });
}

function groupLabel(index: number) {
  return String.fromCharCode("A".charCodeAt(0) + index);
}

function normalizeEnteredId(value: string) {
  const raw = value.trim();
  const numeric = raw.match(/^\d+$/);
  if (numeric) return Number(raw);
  const submission = raw.match(/^S0*(\d+)$/i);
  if (submission) return Number(submission[1]);
  return raw;
}

export default function Page() {
  const [enteredId, setEnteredId] = useState("");
  const [loadedId, setLoadedId] = useState("");
  const [status, setStatus] = useState("");
  const [rankError, setRankError] = useState("");

  const participant = useMemo(() => {
    const normalized = normalizeEnteredId(loadedId);
    return surveyData.find((p) => {
      return (
        Number(p.participant_id) === Number(normalized) ||
        String(p.submission_id).toUpperCase() === loadedId.trim().toUpperCase()
      );
    });
  }, [loadedId]);

  const groups = useMemo(() => {
    if (!participant) return [];
    const orderedMethods = seededOrder(participant.submission_id, methods);
    return orderedMethods.map((method, index) => ({
      label: groupLabel(index),
      method,
      movies: participant.groups[method],
    }));
  }, [participant]);

  function loadParticipant() {
    setStatus("");
    setRankError("");
    setLoadedId(enteredId.trim());
  }

  async function submitSurvey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setRankError("");
    setStatus("");

    if (!participant) return;

    const formData = new FormData(event.currentTarget);
    const ranks = groups.map((g) => String(formData.get(`rank_${g.label}`) || ""));
    if (ranks.some((r) => !r) || new Set(ranks).size !== 4) {
      setRankError("Please use each rank once: 1, 2, 3, and 4.");
      return;
    }

    const payload = {
      entered_id: enteredId,
      participant_id: participant.participant_id,
      submission_id: participant.submission_id,
      respondent: participant.respondent,
      input_movies: participant.input_movies,
      comments: String(formData.get("comments") || ""),
      browser_timestamp: new Date().toISOString(),
      groups: groups.map((group) => ({
        label: group.label,
        method: group.method,
        novelty: String(formData.get(`novelty_${group.label}`) || ""),
        rank: String(formData.get(`rank_${group.label}`) || ""),
        movies: group.movies,
        movie_ratings: group.movies.map((movie, index) => ({
          position: index + 1,
          title: movie,
          relevance: String(formData.get(`relevance_${group.label}_${index + 1}`) || ""),
          would_watch: String(formData.get(`watch_${group.label}_${index + 1}`) || ""),
          seen_before: String(formData.get(`seen_${group.label}_${index + 1}`) || ""),
        })),
      })),
    };

    setStatus("Saving...");

    const response = await fetch("/api/submit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok) {
      setStatus(`Save failed: ${result.error || "Unknown error"}`);
      return;
    }

    setStatus("Saved. Thank you!");
    (event.target as HTMLFormElement).reset();
  }

  return (
    <main className="container">
      <section className="hero">
        <h1>Movie Recommendation Survey</h1>
        <div>Enter your participant ID, review the recommendation groups, and answer the questions.</div>
      </section>

      <section className="card">
        <h2>Start</h2>
        <div className="row">
          <div>
            <label htmlFor="participantId">Participant ID</label>
            <input
              id="participantId"
              value={enteredId}
              onChange={(e) => setEnteredId(e.target.value)}
              placeholder="1–15 or S001–S015"
            />
          </div>
          <button type="button" onClick={loadParticipant}>Load recommendations</button>
        </div>
        <p className="muted">You can enter either a numeric ID, for example 1, or a submission ID, for example S001.</p>
        {loadedId && !participant && <div className="error">Participant ID not found.</div>}
      </section>

      {participant && (
        <form onSubmit={submitSurvey}>
          <section className="card">
            <h2>Participant {participant.participant_id}</h2>
            <div className="identity">Submission: {participant.submission_id} | Name: {participant.respondent}</div>
            <p className="muted">Your original selected movies:</p>
            <div className="inputMovies">{participant.input_movies}</div>
            <p className="muted">All ratings are required. 1 = low / not relevant, 5 = high / very relevant.</p>
          </section>

          {groups.map((group) => (
            <section className="card group" key={group.label}>
              <h2>Recommendation Group {group.label}</h2>
              <div className="grid">
                <div>
                  <label>How interesting or non-obvious is this group?</label>
                  <select name={`novelty_${group.label}`} required>
                    <option value="">Choose</option>
                    {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>
              </div>

              {group.movies.map((movie, index) => (
                <div className="movie" key={movie}>
                  <div className="movieTitle">
                    {index + 1}. <a href={imdbSearchUrl(movie)} target="_blank" rel="noreferrer">{movie}</a>
                  </div>
                  <div className="grid">
                    <div>
                      <label>Relevance to your taste</label>
                      <select name={`relevance_${group.label}_${index + 1}`} required>
                        <option value="">Choose</option>
                        {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </div>
                    <div>
                      <label>How likely are you to watch/enjoy it?</label>
                      <select name={`watch_${group.label}_${index + 1}`} required>
                        <option value="">Choose</option>
                        {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </div>
                    <div>
                      <label>Have you already seen it?</label>
                      <select name={`seen_${group.label}_${index + 1}`} required>
                        <option value="">Choose</option>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="Not sure">Not sure</option>
                      </select>
                    </div>
                  </div>
                </div>
              ))}
            </section>
          ))}

          <section className="card">
            <h2>Overall ranking</h2>
            <p className="muted">Rank the four groups from best to worst. Use each rank once.</p>
            <div className="rankGrid">
              {groups.map((group) => (
                <div key={group.label}>
                  <label>Group {group.label}</label>
                  <select name={`rank_${group.label}`} required>
                    <option value="">Choose</option>
                    <option value="1">1 - best</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4 - worst</option>
                  </select>
                </div>
              ))}
            </div>
            {rankError && <div className="error">{rankError}</div>}
          </section>

          <section className="card">
            <label htmlFor="comments">Optional comments</label>
            <textarea id="comments" name="comments" placeholder="Anything you liked, disliked, or found surprising?" />
          </section>

          <section className="card">
            <button type="submit">Submit answers</button>
            {status && <span className={status.startsWith("Saved") ? "success" : status.startsWith("Save failed") ? "error" : "muted"}> {status}</span>}
          </section>
        </form>
      )}
    </main>
  );
}
