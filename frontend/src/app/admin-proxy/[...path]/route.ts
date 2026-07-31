import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.KAGYA_BACKEND_URL ?? "http://127.0.0.1:8000";
const METHODS = new Set(["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]);

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function HEAD(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

async function proxyRequest(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  const { path } = await context.params;
  const backendPath = path.join("/");
  if (!METHODS.has(request.method) || !isSafeBackendPath(backendPath)) {
    return NextResponse.json({ detail: "Invalid proxy path" }, { status: 400 });
  }

  const target = new URL(`/api/${backendPath}`, BACKEND_URL);
  target.search = request.nextUrl.search;
  const response = await fetch(target, {
    method: request.method,
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
    },
    body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
  });

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: {
      "Content-Type": response.headers.get("Content-Type") ?? "application/json",
    },
  });
}

function isSafeBackendPath(path: string): boolean {
  if (!path) return false;
  if (path.includes("\\") || /[\0-\x1f\x7f]/.test(path)) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("//")) return false;
  let decoded = path;
  for (let index = 0; index < 3; index += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      return false;
    }
  }
  if (decoded.includes("\\") || /[\0-\x1f\x7f]/.test(decoded)) return false;
  if (/^[a-z][a-z0-9+.-]*:/i.test(decoded) || decoded.startsWith("//")) return false;
  return decoded.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
}
