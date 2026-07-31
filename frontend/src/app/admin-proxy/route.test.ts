import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("private backend proxy", () => {
  it("forwards GET requests with query strings", async () => {
    const backendFetch = vi.fn().mockResolvedValue(Response.json({ ok: true }));
    vi.stubGlobal("fetch", backendFetch);
    vi.resetModules();
    const { GET } = await import("./[...path]/route");

    const response = await GET(
      new NextRequest("http://localhost/admin-proxy/system/events?limit=5"),
      { params: Promise.resolve({ path: ["system", "events"] }) },
    );

    expect(response.status).toBe(200);
    expect(backendFetch).toHaveBeenCalledWith(
      new URL("http://127.0.0.1:8000/api/system/events?limit=5"),
      expect.objectContaining({ method: "GET", body: undefined }),
    );
  });

  it.each(["POST", "PUT", "PATCH", "DELETE"])("forwards %s request bodies", async (method) => {
    const backendFetch = vi.fn().mockResolvedValue(new Response("accepted", { status: 202, headers: { "Content-Type": "text/plain" } }));
    vi.stubGlobal("fetch", backendFetch);
    vi.resetModules();
    const route = await import("./[...path]/route");

    const response = await route[method as "POST"](
      new NextRequest("http://localhost/admin-proxy/motivation/reevaluate", {
        method,
        body: JSON.stringify({ ok: true }),
        headers: { "Content-Type": "application/vnd.kagya+json" },
      }),
      { params: Promise.resolve({ path: ["motivation", "reevaluate"] }) },
    );

    expect(response.status).toBe(202);
    expect(response.headers.get("Content-Type")).toBe("text/plain");
    expect(backendFetch).toHaveBeenCalledWith(
      expect.any(URL),
      expect.objectContaining({
        method,
        body: JSON.stringify({ ok: true }),
        headers: { "Content-Type": "application/vnd.kagya+json" },
      }),
    );
  });

  it("defaults missing backend content-type to JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));
    vi.resetModules();
    const { GET } = await import("./[...path]/route");

    const response = await GET(new NextRequest("http://localhost/admin-proxy/goals"), { params: Promise.resolve({ path: ["goals"] }) });

    expect(response.status).toBe(204);
    expect(response.headers.get("Content-Type")).toBe("application/json");
  });

  it.each([
    [[]],
    [["."]],
    [[".."]],
    [["state", "..", "system"]],
    [["state%2F..%2Fsystem"]],
    [["state%252F..%252Fsystem"]],
    [["http:", "evil.test"]],
    [["//evil.test"]],
    [["state\\system"]],
    [["state\u0000system"]],
  ])("rejects unsafe path %j", async (path) => {
    const backendFetch = vi.fn();
    vi.stubGlobal("fetch", backendFetch);
    vi.resetModules();
    const { GET } = await import("./[...path]/route");

    const response = await GET(new NextRequest(`http://localhost/admin-proxy/${path.join("/")}`), { params: Promise.resolve({ path }) });

    expect(response.status).toBe(400);
    expect(backendFetch).not.toHaveBeenCalled();
  });
});
