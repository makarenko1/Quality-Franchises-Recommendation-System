// Vercel serverless function — forwards a survey response to the Google Apps
// Script web app. Keeps the Apps Script URL server-side (set APPS_SCRIPT_URL in
// the Vercel project's Environment Variables) and avoids browser CORS issues.

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const url = process.env.APPS_SCRIPT_URL;
  if (!url) {
    res.status(500).json({ error: "APPS_SCRIPT_URL is not configured" });
    return;
  }

  try {
    const body = typeof req.body === "string" ? req.body : JSON.stringify(req.body);
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!upstream.ok) {
      res.status(502).json({ error: "Upstream error", status: upstream.status });
      return;
    }
    res.status(200).json({ result: "success" });
  } catch (e) {
    res.status(500).json({ error: "Failed to forward", detail: String(e) });
  }
}
