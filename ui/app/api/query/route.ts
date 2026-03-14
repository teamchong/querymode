import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  const { code } = await request.json();

  if (!code || typeof code !== "string") {
    return new Response("code is required", { status: 400 });
  }

  try {
    // Forward to querymode engine via service binding
    const env = (process as unknown as { env: { QUERYMODE: Fetcher } }).env;
    const res = await env.QUERYMODE.fetch("https://querymode/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });

    if (!res.ok) {
      return new Response(await res.text(), { status: res.status });
    }

    return new Response(res.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch (e) {
    return Response.json(
      { rows: [], columns: [], rowCount: 0, bytesRead: 0, pagesSkipped: 0, durationMs: 0, error: String(e) },
      { status: 200 },
    );
  }
}
