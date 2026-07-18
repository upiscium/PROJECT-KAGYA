import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.restoreAllMocks();
  delete process.env.KAGYA_ADMIN_AUTH_ENABLED;
  delete process.env.KAGYA_ADMIN_TOKEN;
  delete process.env.KAGYA_SSO_TRUST_TOKEN;
});

describe("admin proxy authentication", () => {
  it("preserves token injection when optional auth is disabled", async () => {
    process.env.KAGYA_ADMIN_TOKEN = "backend-token";
    const backendFetch = vi.fn().mockResolvedValue(Response.json({ ok: true }));
    vi.stubGlobal("fetch", backendFetch);
    vi.resetModules();
    const { POST } = await import("./[...path]/route");

    const response = await POST(
      new NextRequest("http://localhost/admin-proxy/motivation/reevaluate", {
        method: "POST",
        body: "{}",
      }),
      { params: Promise.resolve({ path: ["motivation", "reevaluate"] }) },
    );

    expect(response.status).toBe(200);
    expect(backendFetch).toHaveBeenCalledWith(
      expect.any(URL),
      expect.objectContaining({
        headers: expect.objectContaining({ "X-KAGYA-Admin-Token": "backend-token" }),
      }),
    );
  });

  it("requires a trusted SSO assertion before issuing strict session cookies", async () => {
    configureAuth();
    vi.resetModules();
    const { GET } = await import("./[...path]/route");
    const context = { params: Promise.resolve({ path: ["auth", "session"] }) };

    const rejected = await GET(
      new NextRequest("https://kagya.example/admin-proxy/auth/session"),
      context,
    );
    const accepted = await GET(sessionRequest("approval_only"), context);

    expect(rejected.status).toBe(401);
    expect(accepted.status).toBe(200);
    expect(accepted.cookies.get("kagya_admin_session")?.httpOnly).toBe(true);
    expect(accepted.cookies.get("kagya_admin_session")?.sameSite).toBe("strict");
    expect(accepted.cookies.get("kagya_admin_csrf")?.sameSite).toBe("strict");
  });

  it("rejects cross-site and role-forbidden mutations before backend fetch", async () => {
    configureAuth();
    const backendFetch = vi.fn().mockResolvedValue(Response.json({ ok: true }));
    vi.stubGlobal("fetch", backendFetch);
    vi.resetModules();
    const { GET, POST } = await import("./[...path]/route");
    const sessionResponse = await GET(sessionRequest("approval_only"), {
      params: Promise.resolve({ path: ["auth", "session"] }),
    });
    const body = await sessionResponse.json() as { csrfToken: string };
    const cookie = sessionResponse.cookies.get("kagya_admin_session")?.value;
    const csrfCookie = sessionResponse.cookies.get("kagya_admin_csrf")?.value;
    const context = { params: Promise.resolve({ path: ["motivation", "reevaluate"] }) };

    const forbidden = await POST(
      mutationRequest(cookie, csrfCookie, body.csrfToken, "same-origin"),
      context,
    );
    const crossSite = await POST(
      mutationRequest(cookie, csrfCookie, body.csrfToken, "cross-site"),
      context,
    );

    expect(forbidden.status).toBe(403);
    expect(crossSite.status).toBe(403);
    expect(backendFetch).not.toHaveBeenCalled();
  });
});

function configureAuth(): void {
  process.env.KAGYA_ADMIN_TOKEN = "backend-token";
  process.env.KAGYA_ADMIN_AUTH_ENABLED = "true";
  process.env.KAGYA_SSO_TRUST_TOKEN = "proxy-secret";
}

function sessionRequest(role: string): NextRequest {
  return new NextRequest("https://kagya.example/admin-proxy/auth/session", {
    headers: {
      "x-kagya-sso-secret": "proxy-secret",
      "x-forwarded-user": "alice@example.test",
      "x-kagya-role": role,
    },
  });
}

function mutationRequest(
  session: string | undefined,
  csrfCookie: string | undefined,
  csrfHeader: string,
  fetchSite: string,
): NextRequest {
  return new NextRequest("https://kagya.example/admin-proxy/motivation/reevaluate", {
    method: "POST",
    body: "{}",
    headers: {
      origin: "https://kagya.example",
      "sec-fetch-site": fetchSite,
      "x-kagya-csrf-token": csrfHeader,
      cookie: `kagya_admin_session=${session}; kagya_admin_csrf=${csrfCookie}`,
    },
  });
}
