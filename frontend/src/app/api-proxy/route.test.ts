import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("public API proxy", () => {
  it("preserves SSE anti-buffering headers", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("event: status\n\n", {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
      },
    })));
    vi.resetModules();
    const { GET } = await import("./[...path]/route");

    const response = await GET(
      new NextRequest("http://localhost/api-proxy/chat/jobs/job-1/events"),
      { params: Promise.resolve({ path: ["chat", "jobs", "job-1", "events"] }) },
    );

    expect(response.headers.get("Content-Type")).toContain("text/event-stream");
    expect(response.headers.get("X-Accel-Buffering")).toBe("no");
  });
});
