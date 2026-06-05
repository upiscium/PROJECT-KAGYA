import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ChatClient } from "./chat-client";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

function renderWithQuery() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}><ChatClient /></QueryClientProvider>);
}

describe("ChatClient", () => {
  it("renders normal chat response without hidden thought fields", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        response: "Visible answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null },
      }),
    });
    renderWithQuery();

    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Visible answer")).toBeInTheDocument();
    expect(screen.queryByText(/hidden_thought/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/raw prompt/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/retrieved memory/i)).not.toBeInTheDocument();
  });

  it("sends selected attachments with the chat request", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        episode_id: "episode-1",
        response: "Visible answer",
        emotion: { valence: 0.1, arousal: 0.2, optimal_loss: 0.9 },
        model: { model_id: "google/gemma-4-E4B", adapter_id: null },
      }),
    });
    renderWithQuery();

    await userEvent.selectOptions(screen.getByLabelText("Attachment type"), "audio");
    await userEvent.type(screen.getByPlaceholderText("Attachment URL"), "file:///tmp/sample.wav");
    await userEvent.type(screen.getByPlaceholderText("Optional attachment name"), "sample.wav");
    await userEvent.click(screen.getByRole("button", { name: "Add Attachment" }));
    await userEvent.type(screen.getByPlaceholderText("Send a message to PROJECT-KAGYA"), "please inspect this");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    const request = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(fetchMock.mock.calls[0][0]).toBe("/api-proxy/chat");
    expect(request).toEqual({
      text: "please inspect this",
      attachments: [{ type: "audio", url: "file:///tmp/sample.wav", name: "sample.wav" }],
      debug: false,
    });
    expect(await screen.findByText(/Attachments: sample.wav/)).toBeInTheDocument();
  });
});
