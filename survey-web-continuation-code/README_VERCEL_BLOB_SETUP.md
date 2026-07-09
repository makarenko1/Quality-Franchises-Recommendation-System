# Vercel Movie Recommendation Survey App — Blob Version

Use this version if your Vercel Storage screen shows **Blob** but not KV.

This version does **not** use Google Cloud.

Participants use the public Vercel link. Each submission is saved as a small JSON file in Vercel Blob storage. The export endpoint combines all saved JSON files into one CSV.

## Setup in Vercel

1. Push this project to GitHub.
2. Import/deploy it in Vercel.
3. In Vercel, open the project.
4. Go to **Storage**.
5. Choose **Blob**.
6. Create/connect a Blob store to this project.
7. Vercel will add `BLOB_READ_WRITE_TOKEN` automatically.
8. Redeploy the project.

## Optional export password

In Vercel Environment Variables, add:

```text
EXPORT_TOKEN=choose-a-private-password
```

Then redeploy.

## Download results as CSV

If you set `EXPORT_TOKEN`, open:

```text
https://YOUR-VERCEL-APP.vercel.app/api/export?token=YOUR_EXPORT_TOKEN
```

If you did not set `EXPORT_TOKEN`, open:

```text
https://YOUR-VERCEL-APP.vercel.app/api/export
```

It downloads:

```text
survey_responses.csv
```

## How saving works

- Submit creates one file under `responses/` in Vercel Blob.
- Export reads all files under `responses/` and combines them into one CSV.
- This avoids trying to append to a local CSV file, which does not work reliably on Vercel.

## Saved format

The CSV is long-format: one row per recommended movie, including participant ID, submission ID, group label, hidden method, group rank, novelty, movie title, relevance, would-watch score, seen-before answer, and comments.


## TypeScript build fix

This package includes an explicit type annotation in `app/api/export/route.ts` for the Vercel Blob `list()` result. This fixes the strict TypeScript build error about `result` implicitly having type `any`.
