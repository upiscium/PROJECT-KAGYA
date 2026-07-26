import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.KAGYA_BACKEND_URL ?? "http://127.0.0.1:8000";

type RouteContext = { params: Promise<{ path: string[] }> };

const ALLOWED_ROUTES = [
  { method: "POST", pattern: /^chat$/ },
  { method: "POST", pattern: /^chat\/jobs$/ },
  { method: "GET", pattern: /^chat\/jobs\/[A-Za-z0-9-]+(?:\/result|\/events)?$/ },
  { method: "DELETE", pattern: /^chat\/jobs\/[A-Za-z0-9-]+$/ },
];

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const backendPath = path.join("/");
  if (!isAllowed(request.method, backendPath)) {
    return NextResponse.json({ detail: "API proxy route is not allowed" }, { status: 404 });
  }

  const target = new URL(`/api/${backendPath}`, BACKEND_URL);
  target.search = request.nextUrl.search;
  let response: Response;
  try {
    response = await fetch(target, {
      method: request.method,
      headers: forwardHeaders(request),
      body: request.method === "GET" ? undefined : await request.text(),
    });
  } catch {
    return NextResponse.json({ detail: `Backend API is not reachable at ${BACKEND_URL}` }, { status: 502 });
  }

  const headers = new Headers({
    "Content-Type": response.headers.get("Content-Type") ?? "application/json",
    "Cache-Control": response.headers.get("Cache-Control") ?? "no-cache",
  });
  const buffering = response.headers.get("X-Accel-Buffering");
  if (buffering) headers.set("X-Accel-Buffering", buffering);
  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function forwardHeaders(request: NextRequest): Headers {
  const headers = new Headers({ "Content-Type": request.headers.get("Content-Type") ?? "application/json" });
  for (const name of ["Idempotency-Key", "X-KAGYA-Client-ID", "Last-Event-ID"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

function isAllowed(method: string, path: string): boolean {
  return ALLOWED_ROUTES.some((route) => route.method === method && route.pattern.test(path));
}
