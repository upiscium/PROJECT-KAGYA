import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { OutboxClient } from "./outbox-client";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: { outboxMessages: vi.fn(), deliverOutbox: vi.fn(), respondToOutbox: vi.fn() },
}));

const mockedApi = vi.mocked(api);

describe("OutboxClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.outboxMessages.mockResolvedValue({ messages: [{
      message_id: "message/1", kind: "approval_request", urgency: "high", delivery_status: "delivered",
      acknowledgment_status: "unacknowledged", title: "Approve deployment", created_at: "2026-01-01T00:00:00Z",
      channel: "operator", privacy_class: "safe", last_failure_code: null,
      references: { action_id: "action/1", decision_id: null, event_id: null, goal_id: null, plan_id: null, commitment_id: null },
    } as never] });
  });

  it("links approval requests to cockpit without rendering approval controls or body", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><OutboxClient /></QueryClientProvider>);
    const link = await screen.findByRole("link", { name: "Open in Cockpit" });
    expect(link).toHaveAttribute("href", "/cockpit#action-action%2F1");
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  });
});
