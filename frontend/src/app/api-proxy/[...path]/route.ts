import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.KAGYA_BACKEND_URL ?? "http://127.0.0.1:8000";

type RouteContext = { params: Promise<{ path: string[] }> };

const ALLOWED_ROUTES = [{ method: "POST", pattern: /^chat$/ }];

export async function POST(request: NextRequest, context: RouteContext) {
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
      headers: { "Content-Type": request.headers.get("Content-Type") ?? "application/json" },
      body: await request.text(),
    });
  } catch {
    return NextResponse.json({ detail: `Backend API is not reachable at ${BACKEND_URL}` }, { status: 502 });
  }

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}

function isAllowed(method: string, path: string): boolean {
  return ALLOWED_ROUTES.some((route) => route.method === method && route.pattern.test(path));
}
