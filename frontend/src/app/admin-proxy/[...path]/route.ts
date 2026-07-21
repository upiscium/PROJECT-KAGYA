import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.KAGYA_BACKEND_URL ?? "http://127.0.0.1:8000";
const ADMIN_TOKEN = process.env.KAGYA_ADMIN_TOKEN;
const AUTH_ENABLED = process.env.KAGYA_ADMIN_AUTH_ENABLED === "true";
const SSO_TRUST_TOKEN = process.env.KAGYA_SSO_TRUST_TOKEN;
const SSO_TRUST_HEADER = process.env.KAGYA_SSO_TRUST_HEADER ?? "x-kagya-sso-secret";
const SSO_ACTOR_HEADER = process.env.KAGYA_SSO_ACTOR_HEADER ?? "x-forwarded-user";
const SSO_ROLE_HEADER = process.env.KAGYA_SSO_ROLE_HEADER ?? "x-kagya-role";
const SSO_REAUTH_HEADER = process.env.KAGYA_SSO_REAUTH_HEADER ?? "x-kagya-reauthenticated-at";
const SESSION_COOKIE = process.env.KAGYA_ADMIN_SESSION_COOKIE ?? "kagya_admin_session";
const CSRF_COOKIE = process.env.KAGYA_ADMIN_CSRF_COOKIE ?? "kagya_admin_csrf";
const BACKEND_ACTOR_HEADER = process.env.KAGYA_BACKEND_ACTOR_HEADER ?? "X-KAGYA-Actor";
const BACKEND_ROLE_HEADER = process.env.KAGYA_BACKEND_ROLE_HEADER ?? "X-KAGYA-Role";
const BACKEND_REAUTH_HEADER = process.env.KAGYA_BACKEND_REAUTH_HEADER ?? "X-KAGYA-Reauthenticated-At";
const CSRF_HEADER = process.env.KAGYA_ADMIN_CSRF_HEADER ?? "x-kagya-csrf-token";
const SESSION_MAX_AGE_SECONDS = Number(process.env.KAGYA_ADMIN_SESSION_MAX_AGE_SECONDS ?? 28800);

type RouteContext = { params: Promise<{ path: string[] }> };
type AdminRole = "read_only" | "approval_only" | "full_admin";
type AdminSession = { actor: string; role: AdminRole; issuedAt: number; reauthenticatedAt: number | null };

const ALLOWED_ROUTES = [
  { method: "GET", pattern: /^state\/emotion$/ },
  { method: "GET", pattern: /^memory\/search$/ },
  { method: "GET", pattern: /^memory\/episodes\/[^/]+$/ },
  { method: "GET", pattern: /^memory\/semantic\/[^/]+(?:\/graph)?$/ },
  { method: "POST", pattern: /^memory\/episodes\/[^/]+\/(archive|metadata)$/ },
  { method: "POST", pattern: /^memory\/semantic\/[^/]+\/(archive|metadata|lifecycle|relationships|policy)$/ },
  { method: "DELETE", pattern: /^memory\/semantic\/[^/]+$/ },
  { method: "POST", pattern: /^chat\/debug$/ },
  { method: "POST", pattern: /^sleep\/run$/ },
  { method: "GET", pattern: /^adapters$/ },
  { method: "GET", pattern: /^adapters\/[^/]+\/provenance$/ },
  { method: "POST", pattern: /^adapters\/[^/]+\/(evaluate|trial|approve|activate|reject)$/ },
  { method: "POST", pattern: /^adapters\/[^/]+\/canary$/ },
  { method: "GET", pattern: /^evaluations$/ },
  { method: "GET", pattern: /^evaluations\/adapters\/[^/]+\/history$/ },
  { method: "GET", pattern: /^evaluations\/[^/]+\.json$/ },
  { method: "GET", pattern: /^system\/(events|journal)$/ },
  { method: "GET", pattern: /^experiences(?:\/[^/]+)?$/ },
  { method: "GET", pattern: /^beliefs$/ },
  { method: "POST", pattern: /^beliefs(?:\/[^/]+\/(resolve|retract|supersede)|\/expire)?$/ },
  { method: "GET", pattern: /^motivation$/ },
  { method: "POST", pattern: /^motivation\/(reevaluate|decay)$/ },
];

const APPROVAL_ROUTES = [
  /^adapters\/[^/]+\/(approve|reject)$/,
  /^beliefs\/[^/]+\/(resolve|retract|supersede)$/,
];

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxyAdminRequest(request, context);
}

async function proxyAdminRequest(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  if (!ADMIN_TOKEN) {
    return NextResponse.json({ detail: "KAGYA_ADMIN_TOKEN is not configured" }, { status: 503 });
  }
  const { path } = await context.params;
  const backendPath = path.join("/");
  if (AUTH_ENABLED && backendPath === "auth/session" && request.method === "GET") {
    return establishSession(request);
  }
  if (!isAllowed(request.method, backendPath)) {
    return NextResponse.json({ detail: "Admin proxy route is not allowed" }, { status: 404 });
  }

  const session = AUTH_ENABLED ? readSession(request) : null;
  if (AUTH_ENABLED && session === null) {
    return NextResponse.json({ detail: "Admin session is required" }, { status: 401 });
  }
  if (session && !roleAllows(session.role, request.method, backendPath)) {
    return NextResponse.json({ detail: "Admin role does not permit this operation" }, { status: 403 });
  }
  if (session && request.method !== "GET") {
    const protectionError = validateBrowserMutation(request);
    if (protectionError) return protectionError;
  }

  const target = new URL(`/api/${backendPath}`, BACKEND_URL);
  target.search = request.nextUrl.search;
  const headers: Record<string, string> = {
    "Content-Type": request.headers.get("Content-Type") ?? "application/json",
    "X-KAGYA-Admin-Token": ADMIN_TOKEN,
  };
  if (session) {
    headers[BACKEND_ACTOR_HEADER] = session.actor;
    headers[BACKEND_ROLE_HEADER] = session.role;
    if (session.reauthenticatedAt !== null) {
      headers[BACKEND_REAUTH_HEADER] = String(session.reauthenticatedAt);
    }
    for (const name of ["origin", "sec-fetch-site", CSRF_HEADER, "cookie"]) {
      const value = request.headers.get(name);
      if (value) headers[name] = value;
    }
  }
  const response = await fetch(target, {
    method: request.method,
    headers,
    body: request.method === "GET" ? undefined : await request.text(),
  });

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: { "Content-Type": response.headers.get("Content-Type") ?? "application/json" },
  });
}

function establishSession(request: NextRequest): NextResponse {
  if (!SSO_TRUST_TOKEN) {
    return NextResponse.json({ detail: "KAGYA_SSO_TRUST_TOKEN is not configured" }, { status: 503 });
  }
  const suppliedTrust = request.headers.get(SSO_TRUST_HEADER);
  if (!suppliedTrust || !safeEqual(suppliedTrust, SSO_TRUST_TOKEN)) {
    return NextResponse.json({ detail: "Trusted SSO assertion is required" }, { status: 401 });
  }
  if (request.headers.get("sec-fetch-site") === "cross-site") {
    return NextResponse.json({ detail: "Cross-site admin request rejected" }, { status: 403 });
  }
  const actor = request.headers.get(SSO_ACTOR_HEADER)?.trim();
  const role = parseRole(request.headers.get(SSO_ROLE_HEADER));
  if (!actor || actor.length > 128 || role === null) {
    return NextResponse.json({ detail: "Valid SSO actor and role are required" }, { status: 401 });
  }
  const now = Math.floor(Date.now() / 1000);
  const reauthenticatedAt = parseTimestamp(request.headers.get(SSO_REAUTH_HEADER));
  const session = signSession({ actor, role, issuedAt: now, reauthenticatedAt });
  const csrf = randomBytes(32).toString("base64url");
  const response = NextResponse.json({ enabled: true, actor, role, csrfToken: csrf });
  response.headers.set("Cache-Control", "no-store");
  const secure = request.nextUrl.protocol === "https:";
  response.cookies.set(SESSION_COOKIE, session, { httpOnly: true, sameSite: "strict", secure, path: "/" });
  response.cookies.set(CSRF_COOKIE, csrf, { httpOnly: false, sameSite: "strict", secure, path: "/" });
  return response;
}

function readSession(request: NextRequest): AdminSession | null {
  const value = request.cookies.get(SESSION_COOKIE)?.value;
  if (!value || !ADMIN_TOKEN) return null;
  const separator = value.lastIndexOf(".");
  if (separator < 1) return null;
  const encoded = value.slice(0, separator);
  if (!safeEqual(value.slice(separator + 1), sessionSignature(encoded))) return null;
  try {
    const session = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8")) as AdminSession;
    const now = Math.floor(Date.now() / 1000);
    if (!session.actor || parseRole(session.role) === null || now - session.issuedAt > SESSION_MAX_AGE_SECONDS || session.issuedAt > now + 30) return null;
    return session;
  } catch {
    return null;
  }
}

function validateBrowserMutation(request: NextRequest): NextResponse | null {
  const origin = request.headers.get("origin");
  const allowed = new Set((process.env.KAGYA_ADMIN_ALLOWED_ORIGINS ?? request.nextUrl.origin).split(",").map((value) => value.trim()));
  if (!origin || !allowed.has(origin) || request.headers.get("sec-fetch-site") === "cross-site") {
    return NextResponse.json({ detail: "Cross-site admin request rejected" }, { status: 403 });
  }
  const cookieToken = request.cookies.get(CSRF_COOKIE)?.value;
  const headerToken = request.headers.get(CSRF_HEADER);
  if (!cookieToken || !headerToken || !safeEqual(cookieToken, headerToken)) {
    return NextResponse.json({ detail: "Invalid admin CSRF token" }, { status: 403 });
  }
  return null;
}

function signSession(session: AdminSession): string {
  const encoded = Buffer.from(JSON.stringify(session)).toString("base64url");
  return `${encoded}.${sessionSignature(encoded)}`;
}

function sessionSignature(encoded: string): string {
  return createHmac("sha256", ADMIN_TOKEN ?? "").update(encoded).digest("base64url");
}

function safeEqual(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

function parseRole(value: string | null): AdminRole | null {
  const normalized = value?.trim().toLowerCase().replaceAll("-", "_");
  return normalized === "read_only" || normalized === "approval_only" || normalized === "full_admin" ? normalized : null;
}

function parseTimestamp(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function roleAllows(role: AdminRole, method: string, path: string): boolean {
  if (role === "full_admin" || method === "GET") return true;
  return role === "approval_only" && APPROVAL_ROUTES.some((pattern) => pattern.test(path));
}

function isAllowed(method: string, path: string): boolean {
  return ALLOWED_ROUTES.some((route) => route.method === method && route.pattern.test(path));
}
