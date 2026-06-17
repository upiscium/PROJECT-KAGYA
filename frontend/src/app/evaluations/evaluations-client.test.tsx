import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationsClient } from "./evaluations-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><EvaluationsClient /></QueryClientProvider>);
}

describe("EvaluationsClient", () => {
  it("lists evaluation results and loads selected JSON", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (url === "/admin-proxy/evaluations") {
        return {
          ok: true,
          json: async () => ({
            results: [
              {
                filename: "adapter-a.json",
                adapter_id: "adapter-a",
                score: 0.9,
                decision: "trial_active",
                case_count: 2,
                updated_at: "2026-06-08T00:00:00+00:00",
              },
              {
                filename: "adapter-b.json",
                adapter_id: "adapter-b",
                score: 0.2,
                decision: "rejected",
                case_count: 1,
                updated_at: "2026-06-07T00:00:00+00:00",
              },
            ],
          }),
        };
      }
      const adapterId = url.endsWith("adapter-b.json") ? "adapter-b" : "adapter-a";
      const decision = adapterId === "adapter-b" ? "rejected" : "trial_active";
      return {
        ok: true,
        json: async () => ({ filename: `${adapterId}.json`, payload: { adapter_id: adapterId, decision } }),
      };
    });

    renderWithQuery();

    expect(await screen.findByText("adapter-a")).toBeInTheDocument();
    expect(screen.getByText("adapter-b")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "View JSON" }));

    expect(await screen.findByText(/"adapter_id": "adapter-b"/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/evaluations", expect.any(Object));
    expect(fetchMock).toHaveBeenCalledWith("/admin-proxy/evaluations/adapter-b.json", expect.any(Object));
  });

  it("shows an empty state when there are no evaluation results", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ results: [] }),
    });

    renderWithQuery();

    expect(await screen.findByText("No evaluation results yet.")).toBeInTheDocument();
    expect(screen.getByText("Select an evaluation result to inspect its JSON payload.")).toBeInTheDocument();
  });
});
