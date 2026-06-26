import { list } from "@vercel/blob";
import { NextResponse } from "next/server";

const headers = [
  "server_timestamp",
  "browser_timestamp",
  "entered_id",
  "participant_id",
  "submission_id",
  "respondent",
  "input_movies",
  "group_label",
  "method",
  "group_rank_1_best",
  "group_novelty_1_5",
  "movie_position",
  "movie_title",
  "relevance_1_5",
  "would_watch_1_5",
  "seen_before",
  "comments",
];

type SurveyRow = Record<string, string>;

function csvEscape(value: unknown) {
  const text = String(value ?? "");
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get("token") || "";

  if (process.env.EXPORT_TOKEN && token !== process.env.EXPORT_TOKEN) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const allRows: SurveyRow[] = [];
  let cursor: string | undefined;

  while (true) {
    const result: Awaited<ReturnType<typeof list>> = await list({
      prefix: "responses/",
      cursor,
    });

    for (const blob of result.blobs) {
      const response = await fetch(blob.url);
      if (!response.ok) continue;

      const rows = (await response.json()) as SurveyRow[];
      if (Array.isArray(rows)) {
        allRows.push(...rows);
      }
    }

    if (!result.cursor) break;
    cursor = result.cursor;
  }

  allRows.sort((a, b) =>
    String(a.server_timestamp).localeCompare(String(b.server_timestamp))
  );

  const csv = [
    headers.join(","),
    ...allRows.map((row) => headers.map((h) => csvEscape(row[h])).join(",")),
  ].join("\n");

  return new Response(csv, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="survey_responses.csv"',
    },
  });
}
