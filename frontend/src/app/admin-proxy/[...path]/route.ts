import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.KAGYA_BACKEND_URL ?? "http://127.0.0.1:8000";
const ADMIN_TOKEN = process.env.KAGYA_ADMIN_TOKEN;

type RouteContext = { params: Promise<{ path: string[] }> };

const ALLOWED_ROUTES = [
  { method: "GET", pattern: /^state\/emotion$/ },
  { method: "GET", pattern: /^memory\/search$/ },
  { method: "GET", pattern: /^memory\/episodes\/[^/]+$/ },
  { method: "GET", pattern: /^memory\/semantic\/[^/]+$/ },
  { method: "POST", pattern: /^memory\/episodes\/[^/]+\/(archive|metadata)$/ },
  { method: "POST", pattern: /^memory\/semantic\/[^/]+\/(archive|metadata)$/ },
  { method: "POST", pattern: /^chat\/debug$/ },
  { method: "POST", pattern: /^sleep\/run$/ },
  { method: "GET", pattern: /^adapters$/ },
  { method: "POST", pattern: /^adapters\/[^/]+\/(evaluate|trial|approve|activate|reject)$/ },
  { method: "GET", pattern: /^evaluations$/ },
  { method: "GET", pattern: /^evaluations\/adapters\/[^/]+\/history$/ },
  { method: "GET", pattern: /^evaluations\/[^/]+\.json$/ },
];

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

async function proxyAdminRequest(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  if (!ADMIN_TOKEN) {
    return NextResponse.json({ detail: "KAGYA_ADMIN_TOKEN is not configured" }, { status: 503 });
  }

  const { path } = await context.params;
  const backendPath = path.join("/");
  if (!isAllowed(request.method, backendPath)) {
    return NextResponse.json({ detail: "Admin proxy route is not allowed" }, { status: 404 });
  }

  const target = new URL(`/api/${backendPath}`, BACKEND_URL);
  target.search = request.nextUrl.search;
  const body = request.method === "GET" ? undefined : await request.text();
  const response = await fetch(target, {
    method: request.method,
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
      "X-KAGYA-Admin-Token": ADMIN_TOKEN,
    },
    body,
  });

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}

function isAllowed(method: string, path: string): boolean {
  return ALLOWED_ROUTES.some((route) => route.method === method && route.pattern.test(path));
}
