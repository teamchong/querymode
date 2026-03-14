import { NextRequest } from "next/server";

export async function GET() {
  try {
    const env = (process as unknown as { env: { QUERYMODE: Fetcher } }).env;
    const res = await env.QUERYMODE.fetch("https://querymode/api/sources");
    return new Response(res.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json([]);
  }
}

export async function POST(request: NextRequest) {
  const env = (process as unknown as { env: { QUERYMODE: Fetcher } }).env;
  const body = await request.arrayBuffer();
  const contentType = request.headers.get("Content-Type") || "application/octet-stream";

  const res = await env.QUERYMODE.fetch("https://querymode/api/sources", {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });

  return new Response(res.body, { status: res.status });
}

export async function DELETE(request: NextRequest) {
  const key = new URL(request.url).searchParams.get("key");
  if (!key) return new Response("key required", { status: 400 });

  const env = (process as unknown as { env: { QUERYMODE: Fetcher } }).env;
  const res = await env.QUERYMODE.fetch(`https://querymode/api/sources?key=${encodeURIComponent(key)}`, {
    method: "DELETE",
  });

  return new Response(res.body, { status: res.status });
}
