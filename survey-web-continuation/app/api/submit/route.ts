import { put } from "@vercel/blob";
import { NextResponse } from "next/server";

type MovieRating = {
  position: number;
  title: string;
  relevance: string;
  would_watch: string;
  seen_before: string;
};

type GroupPayload = {
  label: string;
  method: string;
  novelty: string;
  rank: string;
  movies: string[];
  movie_ratings: MovieRating[];
};

type Payload = {
  entered_id: string;
  participant_id: number;
  submission_id: string;
  respondent: string;
  input_movies: string;
  comments: string;
  browser_timestamp: string;
  groups: GroupPayload[];
};

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as Payload;

    if (!payload || !payload.participant_id || !payload.groups?.length) {
      return NextResponse.json({ error: "Invalid payload." }, { status: 400 });
    }

    const serverTimestamp = new Date().toISOString();
    const rows = [];

    for (const group of payload.groups) {
      for (const movie of group.movie_ratings) {
        rows.push({
          server_timestamp: serverTimestamp,
          browser_timestamp: payload.browser_timestamp || "",
          entered_id: payload.entered_id || "",
          participant_id: String(payload.participant_id),
          submission_id: payload.submission_id || "",
          respondent: payload.respondent || "",
          input_movies: payload.input_movies || "",
          group_label: group.label || "",
          method: group.method || "",
          group_rank_1_best: group.rank || "",
          group_novelty_1_5: group.novelty || "",
          movie_position: String(movie.position || ""),
          movie_title: movie.title || "",
          relevance_1_5: movie.relevance || "",
          would_watch_1_5: movie.would_watch || "",
          seen_before: movie.seen_before || "",
          comments: payload.comments || "",
        });
      }
    }

    if (rows.length === 0) {
      return NextResponse.json({ error: "No rows to save." }, { status: 400 });
    }

    const safeSubmission = String(payload.submission_id || payload.participant_id).replace(/[^a-zA-Z0-9_-]/g, "");
    const filename = `responses/${Date.now()}-${safeSubmission}.json`;

    const blob = await put(filename, JSON.stringify(rows), {
      access: "public",
      contentType: "application/json",
    });

    return NextResponse.json({ ok: true, saved_rows: rows.length, url: blob.url });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || "Save failed." }, { status: 500 });
  }
}
