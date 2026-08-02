import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { evaluationAnchor, evaluationHref } from "@/lib/anchors";
import { CockpitClient } from "./cockpit-client";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      systemInfo: vi.fn(), emotion: vi.fn(), workingMemory: vi.fn(), contexts: vi.fn(), goals: vi.fn(), commitments: vi.fn(), plans: vi.fn(), decisions: vi.fn(), cockpitOutbox: vi.fn(), actionTrace: vi.fn(), actionOperatorSummary: vi.fn(), approveAction: vi.fn(), rejectAction: vi.fn(), cancelAction: vi.fn(), retryAction: vi.fn(), compensateAction: vi.fn(), cockpitTraining: vi.fn(), behavioralEvaluations: vi.fn(), eventJournal: vi.fn(), adapters: vi.fn(), operatorRestoreSummary: vi.fn(), previewOperatorRestore: vi.fn(), commitOperatorRestore: vi.fn(),
    },
  };
});

const mockedApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/cockpit");
  resolveRepresentativeData();
});

describe("CockpitClient", () => {
  it("renders representative safe sections and causal entity links", async () => {
    const { queryClient } = renderCockpit();

    expect(await screen.findByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("report.result")).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "P" && element.textContent?.endsWith("Report result") === true)).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
    expect(screen.getByText("Newest message")).toBeInTheDocument();
    expect(screen.queryByText("Oldest message")).not.toBeInTheDocument();
    expect(screen.getByText("goal_update")).toBeInTheDocument();
    expect(screen.getByText("Action Execution")).toBeInTheDocument();
    const actionMetricsSection = screen.getByText("Action Execution").closest(".ui-card") as HTMLElement;
    expect(within(actionMetricsSection).getByText("Pending approvals").parentElement).toHaveTextContent("2");
    expect(within(actionMetricsSection).getByText("Retry pending").parentElement).toHaveTextContent("1");
    expect(within(actionMetricsSection).getByText("Failed").parentElement).toHaveTextContent("5");
    expect(screen.getByText("Validation rejected")).toBeInTheDocument();
    expect(screen.getByText("Policy rejected")).toBeInTheDocument();
    expect(screen.getByText("awaiting_approval")).toBeInTheDocument();
    expect(screen.getByText("retry_pending")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getAllByText("compensated").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "plan-1" })[0]).toHaveAttribute("href", "#plan-plan-1");
    expect(screen.getAllByRole("link", { name: "decision-1" })[0]).toHaveAttribute("href", "#decision-decision-1");
    expect(screen.getByRole("link", { name: "message-1" })).toHaveAttribute("href", "#outbox-message-1");
    expect(screen.getAllByRole("link", { name: "action-1" })[0]).toHaveAttribute("href", "#action-action-1");
    expect(screen.getByText("Training / Adapters")).toBeInTheDocument();
    expect(screen.getByText("gpu-1")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "job-1" })[0]).toHaveAttribute("href", "#training-job-job-1");
    expect(screen.getAllByRole("link", { name: "adapter-1" })[0]).toHaveAttribute("href", "#adapter-lineage-adapter-1");
    expect(screen.getByRole("link", { name: "eval-1" })).toHaveAttribute("href", "/evaluations#evaluation-eval-1");
    expect(screen.getAllByRole("link", { name: "event-1" })[0]).toHaveAttribute("href", "#journal-event-1");
    const decisionSection = screen.getByText("Recent Decisions").closest(".ui-card") as HTMLElement;
    expect(within(decisionSection).getByRole("link", { name: "action-1" })).toHaveAttribute("href", "#action-action-1");
    expect(within(decisionSection).queryByRole("link", { name: "action-old" })).not.toBeInTheDocument();
    const actionSection = screen.getByText("Action Execution").closest(".ui-card") as HTMLElement;
    const journalSection = screen.getByText("Recent Journal").closest(".ui-card") as HTMLElement;
    const receiptRecord = within(actionSection).getByRole("link", { name: "receipt-1" }).closest(".step-row") as HTMLElement;
    expect(within(receiptRecord).getByRole("link", { name: "receipt-1" })).toHaveAttribute("href", "#receipt-receipt-1");
    expect(within(receiptRecord).getByRole("link", { name: "event-1" })).toHaveAttribute("href", "#journal-event-1");
    expect(within(journalSection).getByRole("link", { name: "action-1" })).toHaveAttribute("href", "#action-action-1");
    expect(within(journalSection).getByRole("link", { name: "receipt-1" })).toHaveAttribute("href", "#receipt-receipt-1");
    expect(screen.getByRole("link", { name: "observation-1" })).toHaveAttribute("href", "#observation-observation-1");
    expect(screen.getByRole("link", { name: "verification-1" })).toHaveAttribute("href", "#verification-verification-1");
    const compensation = screen.getByText((_, element) => element?.tagName === "P" && element.textContent === "Compensates: receipt-original");
    expect(within(compensation).getByRole("link", { name: "receipt-original" })).toHaveAttribute("href", "#receipt-receipt-original");
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
    expect(document.body.textContent).not.toContain("<script>");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "outbox"]))).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "actions"]))).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "action-operator"]))).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "training"]))).not.toContain("PRIVATE_SENTINEL");
  });

  it("targets cockpit evaluation links at the selected evaluation anchor on click", async () => {
    renderCockpit();

    const link = await screen.findByRole("link", { name: "eval-1" });
    link.addEventListener("click", (event) => event.preventDefault());
    await userEvent.click(link);

    expect(link).toHaveAttribute("href", evaluationHref("eval-1"));
    expect(new URL(link.getAttribute("href") ?? "", window.location.origin).hash).toBe(`#${evaluationAnchor("eval-1")}`);
  });

  it("previews only on explicit action and commits the exact preview binding", async () => {
    mockedApi.operatorRestoreSummary.mockResolvedValue({ schema_version: 1, current_sequence: 12, current_snapshot_hash: "c".repeat(64), current_logical_digest: "d".repeat(64), semantic_revision: 4, retained_min_sequence: 1, retained_max_sequence: 12, targets: [{ target_sequence: 7, target_snapshot_hash: "t".repeat(64), checkpoint_kind: "journal_completed", timestamp: "2026-01-01T00:00:00Z", event_type: "goal_update", eligible: true, reason_codes: ["safe_checkpoint"] }], latest_operation: null, external_side_effects_replayed: false });
    const preview = { schema_version: 1, operation_id: "restore-op", preview_digest: "p".repeat(64), created_at: "2026-01-01T00:00:00Z", expires_at: "2026-01-01T01:00:00Z", current_logical_digest: "d".repeat(64), semantic_revision: 4, display_sequence: 12, target_sequence: 7, target_snapshot_hash: "t".repeat(64), newer_authoritative_event_count: 5, domains: [{ domain: "motivation" as const, before_count: 2, after_count: 1, added_count: 0, removed_count: 1, changed_count: 0, changed_revision_count: 0, newer_state_loss_count: 1, refs: [{ kind: "goal" as const, id: "goal-1" }], truncated: false, reason_code: null }], external_effects: { consistency_status: "consistent" as const, artifacts: [], retained_not_replayed_count: 0, pending_count: 0, orphaned_count: 0, retryable_count: 0, effect_digest: "e".repeat(64), external_side_effects_replayed: false }, restoreable: true, reason_codes: ["eligible"], external_side_effects_replayed: false, confirmation_phrase: "RESTORE 7" };
    mockedApi.previewOperatorRestore.mockResolvedValue(preview);
    mockedApi.commitOperatorRestore.mockResolvedValue({ command: "restore", disposition: "completed", operation_id: "restore-op", event_id: "event-restore", processing_sequence: 13, restored_target_sequence: 7, restored_target_hash: "t".repeat(64), post_restore_sequence: 14, post_restore_hash: "r".repeat(64), operation_status: "completed", error_code: null, external_side_effects_replayed: false });
    renderCockpit();
    const target = await screen.findByRole("combobox", { name: "Safe eligible restore target" });
    await userEvent.selectOptions(target, "7");
    expect(mockedApi.previewOperatorRestore).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    await screen.findByText(/Preview digest/);
    const phrase = screen.getByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(phrase, "RESTORE 7");
    await userEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(mockedApi.commitOperatorRestore).toHaveBeenCalledWith({ target_sequence: 7, expected_target_hash: "t".repeat(64), expected_semantic_revision: 4, expected_current_logical_digest: "d".repeat(64), expected_preview_digest: "p".repeat(64), expected_external_effect_digest: "e".repeat(64), confirmation_phrase: "RESTORE 7" });
  });

  it("offers only eligible restore targets and gates Restore on the exact phrase", async () => {
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([
      restoreTarget(7, true),
      restoreTarget(8, false),
      restoreTarget(9, true),
    ]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(restorePreviewFixture() as never);
    renderCockpit();

    const target = await screen.findByRole("combobox", { name: "Safe eligible restore target" });
    expect(within(target).getByRole("option", { name: /Sequence 7/ })).toBeInTheDocument();
    expect(within(target).getByRole("option", { name: /Sequence 9/ })).toBeInTheDocument();
    expect(within(target).queryByRole("option", { name: /Sequence 8/ })).not.toBeInTheDocument();
    await userEvent.selectOptions(target, "7");
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    const restore = await screen.findByRole("button", { name: "Restore" });
    expect(restore).toBeDisabled();
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "WRONG");
    expect(restore).toBeDisabled();
    await userEvent.clear(screen.getByRole("textbox", { name: "Exact restore confirmation" }));
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "RESTORE 7");
    expect(restore).toBeEnabled();
  });

  it("submits duplicate preview and commit clicks only once", async () => {
    const previewRequest = deferred<ReturnType<typeof restorePreviewFixture>>();
    const commitRequest = deferred<unknown>();
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockReturnValue(previewRequest.promise as never);
    mockedApi.commitOperatorRestore.mockReturnValue(commitRequest.promise as never);
    renderCockpit();
    await userEvent.selectOptions(await screen.findByRole("combobox", { name: "Safe eligible restore target" }), "7");
    const previewButton = screen.getByRole("button", { name: "Preview" });
    await vi.waitFor(() => expect(previewButton).toBeEnabled());
    await act(async () => {
      fireEvent.click(previewButton);
      fireEvent.click(previewButton);
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(mockedApi.previewOperatorRestore).toHaveBeenCalledTimes(1));
    await act(async () => {
      previewRequest.resolve(restorePreviewFixture());
      await previewRequest.promise;
    });
    const phrase = await screen.findByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(phrase, "RESTORE 7");
    const restore = screen.getByRole("button", { name: "Restore" });
    await act(async () => {
      fireEvent.click(restore);
      fireEvent.click(restore);
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(mockedApi.commitOperatorRestore).toHaveBeenCalledTimes(1));
    await act(async () => {
      commitRequest.resolve(restoreCommitResponse());
      await commitRequest.promise;
    });
    expect(await screen.findByRole("status")).toHaveTextContent("Restore completed");
  });

  it("renders bounded preview counts and safe anchors without private fields", async () => {
    const preview = restorePreviewFixture();
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(preview as never);
    const { queryClient } = renderCockpit();
    const target = await screen.findByRole("combobox", { name: "Safe eligible restore target" });
    await userEvent.selectOptions(target, "7");
    const previewButton = screen.getByRole("button", { name: "Preview" });
    await vi.waitFor(() => expect(previewButton).toBeEnabled());
    await userEvent.click(previewButton);
    expect(await screen.findByText(/Newer authoritative events: 5/)).toBeInTheDocument();
    expect(screen.getByText(/Public counts: 2 before · 1 after · 0 added · 1 removed · 0 changed · 1 newer-state loss/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "goal:goal-1" })).toHaveAttribute("href", "#goal-goal-1");
    expect(screen.getByText(/External effects: consistent · retained 2 · pending 0 · orphaned 0 · retryable 0/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getQueryCache().getAll().filter((query) => JSON.stringify(query.queryKey).includes("restore")))).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(queryClient.getMutationCache().getAll())).not.toContain("PRIVATE_SENTINEL");
  });

  it("clears a stale preview on 409 and refetches authoritative restore keys", async () => {
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(restorePreviewFixture() as never);
    mockedApi.commitOperatorRestore.mockRejectedValue(new ApiError("stale", 409, null, "restore_preview_stale"));
    const { queryClient } = renderCockpit();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    await userEvent.selectOptions(await screen.findByRole("combobox", { name: "Safe eligible restore target" }), "7");
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    await screen.findByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "RESTORE 7");
    await userEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Restore preview is stale");
    expect(screen.queryByRole("textbox", { name: "Exact restore confirmation" })).not.toBeInTheDocument();
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["cockpit", "restore-summary"] }));
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["cockpit", "restore-preview"] }));
  });

  it("does not retry an indeterminate 503 and displays bounded status", async () => {
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(restorePreviewFixture() as never);
    mockedApi.commitOperatorRestore.mockRejectedValue(new ApiError("indeterminate", 503, null, "commit_indeterminate"));
    renderCockpit();
    await userEvent.selectOptions(await screen.findByRole("combobox", { name: "Safe eligible restore target" }), "7");
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    await screen.findByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "RESTORE 7");
    await userEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(await screen.findByRole("status")).toHaveTextContent("was not retried");
    expect(mockedApi.commitOperatorRestore).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("combobox", { name: "Safe eligible restore target" })).toBeDisabled();
    expect(screen.getByText(/indeterminate and cannot be canceled/)).toBeInTheDocument();
    expect(screen.queryByText(/indeterminate.*indeterminate.*indeterminate/i)).not.toBeInTheDocument();
  });

  it("invalidates cockpit and standalone queries after restore success", async () => {
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(restorePreviewFixture() as never);
    mockedApi.commitOperatorRestore.mockResolvedValue(restoreCommitResponse() as never);
    const standaloneExplanations = vi.fn().mockResolvedValue({ explanations: [{ explanation_id: "fresh" }] });
    const standaloneOutbox = vi.fn().mockResolvedValue({ messages: [{ message_id: "fresh" }] });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    await queryClient.prefetchQuery({ queryKey: ["decision-explanations"], queryFn: standaloneExplanations });
    await queryClient.prefetchQuery({ queryKey: ["outbox"], queryFn: standaloneOutbox });
    const { queryClient: renderedClient } = renderCockpit(queryClient);
    const invalidate = vi.spyOn(renderedClient, "invalidateQueries");
    await userEvent.selectOptions(await screen.findByRole("combobox", { name: "Safe eligible restore target" }), "7");
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    await screen.findByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "RESTORE 7");
    await userEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Restore completed");
    for (const key of ["runtime", "emotion", "working-memory", "contexts", "goals", "commitments", "plans", "decisions", "outbox", "actions", "training", "evaluations", "journal", "adapters", "action-operator", "restore-summary"]) {
      expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["cockpit", key] }));
    }
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["decision-explanations"] }));
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["outbox"] }));
  });

  it("shows pending restore as non-cancelable and locks controls", async () => {
    const commit = deferred<unknown>();
    mockedApi.operatorRestoreSummary.mockResolvedValue(restoreSummaryFixture([restoreTarget(7, true)]) as never);
    mockedApi.previewOperatorRestore.mockResolvedValue(restorePreviewFixture() as never);
    mockedApi.commitOperatorRestore.mockReturnValue(commit.promise as never);
    renderCockpit();
    const target = await screen.findByRole("combobox", { name: "Safe eligible restore target" });
    await userEvent.selectOptions(target, "7");
    await userEvent.click(screen.getByRole("button", { name: "Preview" }));
    await screen.findByRole("textbox", { name: "Exact restore confirmation" });
    await userEvent.type(screen.getByRole("textbox", { name: "Exact restore confirmation" }), "RESTORE 7");
    await userEvent.click(screen.getByRole("button", { name: "Restore" }));
    expect(await screen.findByText("Restore is pending/finalizing or indeterminate and cannot be canceled.")).toBeInTheDocument();
    expect(target).toBeDisabled();
    expect(screen.getByRole("textbox", { name: "Exact restore confirmation" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Restoring…" })).toBeDisabled();
    await act(async () => commit.resolve(restoreCommitResponse()));
  });

  it("keeps existing sections visible when restore summary fails", async () => {
    mockedApi.operatorRestoreSummary.mockRejectedValue(new Error("restore unavailable"));
    renderCockpit();
    expect(await screen.findByText("State replay / restore is unavailable.")).toBeInTheDocument();
    expect(screen.getByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Action Execution")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
  });

  it("shows action loading without hiding loaded sections", async () => {
    mockedApi.actionTrace.mockReturnValue(new Promise(() => undefined));
    renderCockpit();

    expect(await screen.findByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Loading action execution...")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("shows explicit empty states for independent sections", async () => {
    mockedApi.contexts.mockResolvedValue({ contexts: [] });
    mockedApi.goals.mockResolvedValue({ goals: [], decisions: [] });
    mockedApi.commitments.mockResolvedValue({ commitments: [] });
    mockedApi.plans.mockResolvedValue({ plans: [] });
    mockedApi.decisions.mockResolvedValue({ decisions: [] });
    mockedApi.cockpitOutbox.mockResolvedValue({ pending_count: 0, critical_count: 0, messages: [] });
    mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 0, retry_pending_count: 0, failed_count: 0, traces: [], pre_intent_failures: [] });
    mockedApi.cockpitTraining.mockResolvedValue({ node_count: 0, online_node_count: 0, running_job_count: 0, failed_job_count: 0, importing_job_count: 0, active_adapter_count: 0, candidate_adapter_count: 0, nodes: [], jobs: [], adapters: [] });
    mockedApi.eventJournal.mockResolvedValue({ records: [] });
    renderCockpit();

    expect(await screen.findByText("No contexts recorded.")).toBeInTheDocument();
    expect(screen.getByText("No active or candidate goals.")).toBeInTheDocument();
    expect(screen.getByText("No current commitments.")).toBeInTheDocument();
    expect(screen.getByText("No active plans.")).toBeInTheDocument();
    expect(screen.getByText("No decisions recorded.")).toBeInTheDocument();
    expect(screen.getByText("No proactive messages.")).toBeInTheDocument();
    expect(screen.getByText("No action execution traces.")).toBeInTheDocument();
    expect(screen.getByText("No training nodes, jobs, or adapter lineage records.")).toBeInTheDocument();
    expect(screen.getByText("No Journal records.")).toBeInTheDocument();
  });

  it("keeps successful sections visible after an action-only failure", async () => {
    mockedApi.actionTrace.mockRejectedValue(new Error("actions unavailable"));
    renderCockpit();

    expect(await screen.findByText("actions unavailable")).toBeInTheDocument();
    expect(screen.getByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Report results")).toBeInTheDocument();
    expect(screen.getByText("conversation")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
  });

  it("renders server-authorized approval controls and submits preview binding only", async () => {
    mockedApi.actionOperatorSummary.mockResolvedValue(operatorSummary());
    mockedApi.approveAction.mockResolvedValue(operatorMutationResponse("approve") as never);
    renderCockpit();

    expect(await screen.findByText("Approval Inbox")).toBeInTheDocument();
    expect(screen.getByText(/local notification · Public title · Public body/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry now" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(mockedApi.approveAction).toHaveBeenCalledWith("operator-action-1", {
      expected_approval_id: "approval-1",
      expected_intent_revision: 3,
      expected_preview_digest: "a".repeat(64),
    });
  });

  it("displays bounded registry descriptions without registry action controls", async () => {
    mockedApi.actionOperatorSummary.mockResolvedValue({ ...operatorSummary(), registry_tools: [{ name: "registry-only", description: "Reads public metadata.", tool_type: "metadata", status: "declared", generated: false, human_approved: false, execution_authority: "registry_only" }] });
    const { queryClient } = renderCockpit();

    const registryHeading = await screen.findByText("Registry-only tools");
    const registrySection = registryHeading.nextElementSibling as HTMLElement;
    expect(within(registrySection).getByText(/Reads public metadata\./)).toBeInTheDocument();
    expect(within(registrySection).queryAllByRole("button")).toHaveLength(0);
    expect(queryClient.getQueryData(["cockpit", "action-operator"])).toEqual(expect.objectContaining({ registry_tools: [expect.objectContaining({ description: "Reads public metadata." })] }));
  });

  it("keeps action traces visible when operator controls fail", async () => {
    mockedApi.actionOperatorSummary.mockRejectedValue(new Error("operator controls unavailable"));
    renderCockpit();

    expect(await screen.findByText("operator controls unavailable")).toBeInTheDocument();
    expect(screen.getByText("Action Execution")).toBeInTheDocument();
    expect(screen.getAllByText("document_search").length).toBeGreaterThan(0);
  });

  it("keeps successful sections visible after a training-only failure", async () => {
    mockedApi.cockpitTraining.mockRejectedValue(new Error("training unavailable"));
    renderCockpit();

    expect(await screen.findByText("training unavailable")).toBeInTheDocument();
    expect(screen.getByText("Ship safely")).toBeInTheDocument();
    expect(screen.getByText("Action Execution")).toBeInTheDocument();
  });

  it("renders split training nodes with an offline worker without key warnings", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    mockedApi.cockpitTraining.mockResolvedValue({
      ...cockpitTrainingProjection(),
      node_count: 2,
      online_node_count: 1,
      nodes: [
        { node_id: "node-main", role: "inference", backend: "ssh", status: "online", last_contact_at: null, expected_model_id: "model-1", expected_model_revision: "rev-1", expected_processor_revision: "proc-1", observed_model_id: null, observed_model_revision: null, model_matches_expected: null, gpu_name: null, cuda_version: null, driver_version: null },
        { node_id: "training-01", role: "worker", backend: "ssh", status: "unavailable", last_contact_at: null, expected_model_id: "model-1", expected_model_revision: "rev-1", expected_processor_revision: "proc-1", observed_model_id: null, observed_model_revision: null, model_matches_expected: null, gpu_name: null, cuda_version: null, driver_version: null },
      ],
      adapters: [{ ...cockpitTrainingProjection().adapters[0], rollback_event_id: "event-1", rollback_event_sequence: 8 }],
    } as never);

    renderCockpit();

    expect(await screen.findByText("training-01")).toBeInTheDocument();
    expect(screen.getByText("Nodes").parentElement).toHaveTextContent("2");
    expect(screen.getByText("Online").parentElement).toHaveTextContent("1");
    expect(screen.getAllByText("unavailable").length).toBeGreaterThan(0);
    expect(screen.queryByText("unavailable-1")).not.toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "P" && element.textContent?.includes("Evaluation:") === true && element.textContent?.includes("passed") === true)).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "event-1" }).length).toBeGreaterThan(0);
    expect(consoleError.mock.calls.flat().join("\n")).not.toContain("Each child in a list should have a unique");
    consoleError.mockRestore();
  });

  it("links a decision to its pre-intent failure when no intent exists", async () => {
    mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 0, retry_pending_count: 0, failed_count: 2, traces: [], pre_intent_failures: rawActionFailures.map(cockpitFailureProjection) as never });
    const { queryClient } = renderCockpit();

    expect(await screen.findByText("Validation rejected")).toBeInTheDocument();
    expect(screen.getByText("Policy rejected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "validation-1" })).toHaveAttribute("href", "#action-failure-validation-1");
    const actionSection = screen.getByText("Action Execution").closest(".ui-card") as HTMLElement;
    expect(within(actionSection).getByRole("link", { name: "event-1" })).toHaveAttribute("href", "#journal-event-1");
    expect(JSON.stringify(queryClient.getQueryData(["cockpit", "actions"]))).not.toContain("PRIVATE_SENTINEL");
  });

  it("sends approve only, with every server binding, and leaves other commands untouched", async () => {
    mockedApi.actionOperatorSummary.mockResolvedValue(operatorSummary());
    mockedApi.approveAction.mockResolvedValue(operatorMutationResponse("approve") as never);
    renderCockpit();

    const reason = await screen.findByRole("textbox", { name: "Reason for operator-action-1" });
    await userEvent.type(reason, "approved after review");
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(mockedApi.approveAction).toHaveBeenCalledWith("operator-action-1", expect.objectContaining({
      expected_approval_id: "approval-1", expected_intent_revision: 3, expected_preview_digest: "a".repeat(64), reason: "approved after review",
    }));
    expect(mockedApi.rejectAction).not.toHaveBeenCalled();
    expect(mockedApi.cancelAction).not.toHaveBeenCalled();
    expect(mockedApi.retryAction).not.toHaveBeenCalled();
    expect(mockedApi.compensateAction).not.toHaveBeenCalled();
  });

  it("bounds rejection reasons and binds rejection to the approval", async () => {
    mockedApi.actionOperatorSummary.mockResolvedValue(operatorSummary());
    mockedApi.rejectAction.mockResolvedValue(operatorMutationResponse("reject") as never);
    renderCockpit();

    const reason = await screen.findByRole("textbox", { name: "Reason for operator-action-1" });
    await userEvent.type(reason, "x".repeat(700));
    expect(reason).toHaveValue("x".repeat(500));
    await userEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(mockedApi.rejectAction).toHaveBeenCalledWith("operator-action-1", expect.objectContaining({
      reason: "x".repeat(500), expected_approval_id: "approval-1", expected_intent_revision: 3,
      expected_preview_digest: "a".repeat(64),
    }));
  });

  it("waits for the authoritative operator refetch after rejection", async () => {
    const refreshed = { ...operatorSummary(), actions: [], operator_action_count: 0, pending_approval_count: 0 };
    mockedApi.actionOperatorSummary.mockResolvedValueOnce(operatorSummary()).mockResolvedValue(refreshed);
    mockedApi.rejectAction.mockResolvedValue(operatorMutationResponse("reject") as never);
    renderCockpit();
    await userEvent.click(await screen.findByRole("button", { name: "Reject" }));
    await vi.waitFor(() => expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument());
    expect(mockedApi.actionOperatorSummary.mock.calls.length).toBeGreaterThan(1);
  });

  it.each([
    ["cancel", "cancelAction", "cancelled"],
    ["retry_now", "retryAction", "executed"],
    ["compensate", "compensateAction", "compensated"],
  ] as const)("clicks server-authorized %s and binds the request", async (command, method, disposition) => {
    const commands = [command];
    const action = operatorAction({ status: command === "retry_now" ? "retry_pending" : command === "compensate" ? "succeeded" : "approved", available_commands: commands, approval: null, receipt: command === "compensate" ? { receipt_id: "receipt-1", status: "succeeded" } : null });
    mockedApi.actionOperatorSummary.mockResolvedValue({ ...operatorSummary(), actions: [action], operator_action_count: 1 });
    mockedApi[method].mockResolvedValue({ ...operatorMutationResponse("approve"), command, disposition, action } as never);
    renderCockpit();
    const button = await screen.findByRole("button", { name: command === "retry_now" ? "Retry now" : command[0].toUpperCase() + command.slice(1) });
    await userEvent.click(button);
    expect(mockedApi[method]).toHaveBeenCalledWith("operator-action-1", {
      expected_intent_revision: 3, expected_preview_digest: "a".repeat(64),
    });
    for (const other of ["approveAction", "rejectAction", "cancelAction", "retryAction", "compensateAction"] as const) {
      if (other !== method) expect(mockedApi[other]).not.toHaveBeenCalled();
    }
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  });

  it.each([["retry_now", "Retry now"], ["compensate", "Compensate"]] as const)("does not invent omitted %s controls", async (command, label) => {
    mockedApi.actionOperatorSummary.mockResolvedValue({ ...operatorSummary(), actions: [operatorAction({ available_commands: ["cancel"], approval: null })] });
    renderCockpit();
    await screen.findByText("Other operator actions");
    expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    expect(mockedApi[command === "retry_now" ? "retryAction" : "compensateAction"]).not.toHaveBeenCalled();
  });

  it("requires the exact non-null confirmation phrase and sends phrase, intent, revision, and digest", async () => {
    const action = operatorAction({ available_commands: ["cancel"], approval: null, confirmation: { required: true, phrase: "CANCEL OPERATOR-ACTION-1" } });
    mockedApi.actionOperatorSummary.mockResolvedValue({ ...operatorSummary(), actions: [action] });
    mockedApi.cancelAction.mockResolvedValue(operatorMutationResponse("approve") as never);
    renderCockpit();
    const button = await screen.findByRole("button", { name: "Cancel" });
    expect(button).toBeDisabled();
    const phrase = screen.getByRole("textbox", { name: "Confirmation for operator-action-1" });
    await userEvent.type(phrase, "CANCEL OPERATOR-ACTION-1x");
    expect(button).toBeDisabled();
    await userEvent.clear(phrase);
    await userEvent.type(phrase, "CANCEL OPERATOR-ACTION-1");
    expect(button).toBeEnabled();
    await userEvent.click(button);
    expect(mockedApi.cancelAction).toHaveBeenCalledWith("operator-action-1", {
      confirmation_phrase: "CANCEL OPERATOR-ACTION-1", expected_intent_revision: 3, expected_preview_digest: "a".repeat(64),
    });
  });

  it("locks duplicate clicks for one intent while allowing another intent", async () => {
    const first = deferred<unknown>();
    const second = deferred<unknown>();
    const firstAction = operatorAction({ available_commands: ["cancel", "reject"], confirmation: null });
    const secondAction = operatorAction({ intent_id: "operator-action-2", available_commands: ["cancel"], confirmation: null });
    mockedApi.actionOperatorSummary.mockResolvedValue({ ...operatorSummary(), actions: [firstAction, secondAction], operator_action_count: 2 });
    mockedApi.cancelAction.mockReturnValueOnce(first.promise as never).mockReturnValueOnce(second.promise as never);
    renderCockpit();
    const cancelButtons = await screen.findAllByRole("button", { name: "Cancel" });
    fireEvent.click(cancelButtons[0]);
    fireEvent.click(screen.getAllByRole("button", { name: "Reject" })[0]);
    await vi.waitFor(() => expect(mockedApi.cancelAction).toHaveBeenCalledTimes(1));
    expect(mockedApi.rejectAction).not.toHaveBeenCalled();
    expect(screen.getAllByRole("button", { name: "Reject" })[0]).toBeDisabled();
    expect(cancelButtons[1]).toBeEnabled();
    await userEvent.click(cancelButtons[1]);
    await vi.waitFor(() => expect(mockedApi.cancelAction).toHaveBeenCalledTimes(2));
    expect(mockedApi.cancelAction.mock.calls.map(([intentId]) => intentId)).toEqual(["operator-action-1", "operator-action-2"]);
    await act(async () => {
      first.resolve(operatorMutationResponse("approve"));
      second.resolve(operatorMutationResponse("approve"));
    });
  });

  it.each([["success", true], ["stale 409", false]] as const)("refetches all authoritative caches after %s", async (_label, succeeds) => {
    const mutation = deferred<unknown>();
    const stale = operatorAction({ available_commands: ["cancel"], confirmation: null });
    const refreshed = { ...operatorSummary(), actions: [], operator_action_count: 0, pending_approval_count: 0 };
    const refreshedActions = { pending_approval_count: 0, retry_pending_count: 0, failed_count: 0, traces: [], pre_intent_failures: [] };
    const refreshedDecisions = { decisions: [] };
    const refreshedPlans = { plans: [] };
    const refreshedOutbox = { pending_count: 0, critical_count: 0, messages: [] };
    const refreshedJournal = { records: [] };
    mockedApi.actionOperatorSummary.mockResolvedValueOnce({ ...operatorSummary(), actions: [stale] }).mockResolvedValue(refreshed);
    mockedApi.actionTrace.mockResolvedValueOnce({ pending_approval_count: 1, retry_pending_count: 1, failed_count: 1, traces: rawActionFixtures.map(cockpitActionProjection) as never, pre_intent_failures: rawActionFailures.map(cockpitFailureProjection) as never }).mockResolvedValue(refreshedActions);
    mockedApi.decisions.mockResolvedValueOnce({ decisions: [{ decision_id: "decision-stale", context_id: null, active_goal_ids: [], selected_candidate_id: "candidate-stale", selected_candidate: { candidate_id: "candidate-stale", candidate_type: "no_op", proposed_action: "Wait", plan_id: null, plan_revision: null, step_id: null, goal_refs: [], commitment_refs: [] }, selection_confidence: 1, status: "awaiting_outcome", outcome_status: "pending", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] } as never).mockResolvedValue(refreshedDecisions);
    mockedApi.plans.mockResolvedValueOnce({ plans: [{ plan_id: "plan-stale", goal_id: "goal-1", revision: 1, status: "active", steps: [], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] } as never).mockResolvedValue(refreshedPlans);
    mockedApi.cockpitOutbox.mockResolvedValueOnce({ pending_count: 1, critical_count: 1, messages: [cockpitOutboxProjection(rawOutboxFixtures[0])] }).mockResolvedValue(refreshedOutbox);
    mockedApi.eventJournal.mockResolvedValueOnce({ records: [{ record_id: "record-stale", timestamp: "2026-01-01T00:00:00Z", lifecycle: "completed", event_id: "event-stale", event_type: "action_cancel", source: "runtime", processing_sequence: 1, snapshot_sequence: 1, causation_id: null, correlation_id: null, state_hash_before: null, state_hash_after: null, snapshot_hash: null, failure_category: null, actor_id: null, actor_role: null, target: null, reauthenticated: null, previous_record_hash: null, record_hash: "hash" }] } as never).mockResolvedValue(refreshedJournal);
    mockedApi.cancelAction.mockReturnValue(mutation.promise as never);
    const standaloneExplanations = vi.fn().mockResolvedValueOnce({ explanations: [{ explanation_id: "stale" }] }).mockResolvedValue({ explanations: [{ explanation_id: "fresh" }] });
    const standaloneOutbox = vi.fn().mockResolvedValueOnce({ messages: [{ message_id: "stale" }] }).mockResolvedValue({ messages: [{ message_id: "fresh" }] });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    await queryClient.prefetchQuery({ queryKey: ["decision-explanations"], queryFn: standaloneExplanations });
    await queryClient.prefetchQuery({ queryKey: ["outbox"], queryFn: standaloneOutbox });
    const cockpitKeys = ["action-operator", "actions", "decisions", "plans", "outbox", "journal"] as const;
    const { queryClient: renderedClient } = renderCockpit(queryClient);
    const invalidate = vi.spyOn(renderedClient, "invalidateQueries");
    const cancel = await screen.findByRole("button", { name: "Cancel" });
    await vi.waitFor(() => {
      expect(mockedApi.actionTrace).toHaveBeenCalled();
      expect(mockedApi.decisions).toHaveBeenCalled();
      expect(mockedApi.plans).toHaveBeenCalled();
      expect(mockedApi.cockpitOutbox).toHaveBeenCalled();
      expect(mockedApi.eventJournal).toHaveBeenCalled();
    });
    const before = Object.fromEntries(cockpitKeys.map((key) => [key, mockedApi[key === "action-operator" ? "actionOperatorSummary" : key === "actions" ? "actionTrace" : key === "decisions" ? "decisions" : key === "plans" ? "plans" : key === "outbox" ? "cockpitOutbox" : "eventJournal"].mock.calls.length]));
    await userEvent.click(cancel);
    expect(cancel).toBeDisabled();
    expect(renderedClient.getQueryData(["cockpit", "action-operator"])).toEqual(expect.objectContaining({ actions: [expect.objectContaining({ intent_id: "operator-action-1" })] }));
    mutation.promise.catch(() => undefined);
    await act(async () => {
      if (succeeds) mutation.resolve(operatorMutationResponse("approve"));
      else mutation.reject(Object.assign(new Error("Action is stale"), { status: 409 }));
    });
    await vi.waitFor(() => expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument());
    for (const key of cockpitKeys) {
      const method = key === "action-operator" ? "actionOperatorSummary" : key === "actions" ? "actionTrace" : key === "decisions" ? "decisions" : key === "plans" ? "plans" : key === "outbox" ? "cockpitOutbox" : "eventJournal";
      expect(mockedApi[method].mock.calls.length).toBeGreaterThan(before[key]);
      expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["cockpit", key] }));
    }
    expect(standaloneExplanations).toHaveBeenCalledTimes(2);
    expect(standaloneOutbox).toHaveBeenCalledTimes(2);
    expect(renderedClient.getQueryData(["cockpit", "actions"])).toEqual(refreshedActions);
    expect(renderedClient.getQueryData(["cockpit", "decisions"])).toEqual(refreshedDecisions);
    expect(renderedClient.getQueryData(["cockpit", "plans"])).toEqual(refreshedPlans);
    expect(renderedClient.getQueryData(["cockpit", "outbox"])).toEqual(refreshedOutbox);
    expect(renderedClient.getQueryData(["cockpit", "journal"])).toEqual(refreshedJournal);
    expect(renderedClient.getQueryData(["decision-explanations"])).toEqual({ explanations: [{ explanation_id: "fresh" }] });
    expect(renderedClient.getQueryData(["outbox"])).toEqual({ messages: [{ message_id: "fresh" }] });
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["decision-explanations"] }));
    expect(invalidate).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["outbox"] }));
    const actionCache = renderedClient.getQueryCache().getAll().filter((query) => {
      const key = JSON.stringify(query.queryKey);
      return key.includes("action-operator") || key === '["decision-explanations"]' || key === '["outbox"]';
    });
    expect(JSON.stringify(actionCache)).not.toContain("PRIVATE_SENTINEL");
    expect(JSON.stringify(renderedClient.getMutationCache().getAll())).not.toContain("PRIVATE_SENTINEL");
    expect(document.body.textContent).not.toContain("PRIVATE_SENTINEL");
  });
});

function renderCockpit(queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })) {
  return { ...render(<QueryClientProvider client={queryClient}><CockpitClient /></QueryClientProvider>), queryClient };
}

function restoreTarget(sequence: number, eligible: boolean) {
  return { target_sequence: sequence, target_snapshot_hash: String(sequence).repeat(64), checkpoint_kind: "journal_completed" as const, timestamp: "2026-01-01T00:00:00Z", event_type: "goal_update", eligible, reason_codes: eligible ? ["safe_checkpoint"] : ["unsafe_checkpoint"] };
}

function restoreSummaryFixture(targets: ReturnType<typeof restoreTarget>[]) {
  return { schema_version: 1, current_sequence: 12, current_snapshot_hash: "c".repeat(64), current_logical_digest: "d".repeat(64), semantic_revision: 4, retained_min_sequence: 1, retained_max_sequence: 12, targets, latest_operation: null, external_side_effects_replayed: false };
}

function restorePreviewFixture() {
  return {
    schema_version: 1, operation_id: "restore-op", preview_digest: "p".repeat(64), created_at: "2026-01-01T00:00:00Z", expires_at: "2026-01-01T01:00:00Z", current_logical_digest: "d".repeat(64), semantic_revision: 4, display_sequence: 12, target_sequence: 7, target_snapshot_hash: "7".repeat(64), newer_authoritative_event_count: 5,
    domains: [{ domain: "motivation" as const, before_count: 2, after_count: 1, added_count: 0, removed_count: 1, changed_count: 0, changed_revision_count: 0, newer_state_loss_count: 1, refs: [{ kind: "goal" as const, id: "goal-1" }], truncated: false, reason_code: null }],
    external_effects: { consistency_status: "consistent" as const, artifacts: [{ artifact_type: "memory" as const, count: 2, refs: [], truncated: true }], retained_not_replayed_count: 2, pending_count: 0, orphaned_count: 0, retryable_count: 0, effect_digest: "e".repeat(64), external_side_effects_replayed: false },
    restoreable: true, reason_codes: ["eligible"], external_side_effects_replayed: false, confirmation_phrase: "RESTORE 7",
  };
}

function restoreCommitResponse() {
  return { command: "restore" as const, disposition: "completed" as const, operation_id: "restore-op", event_id: "event-restore", processing_sequence: 13, restored_target_sequence: 7, restored_target_hash: "7".repeat(64), post_restore_sequence: 14, post_restore_hash: "r".repeat(64), operation_status: "completed" as const, error_code: null, external_side_effects_replayed: false as const };
}

function resolveRepresentativeData() {
  mockedApi.systemInfo.mockResolvedValue({ project: "KAGYA", status: "ok", build: { version: "1.0", commit: "abc123" }, runtime: { environment: "test", provider: "dummy", primary_model_id: "model-1", fallback_configured: false, transformers_4bit: false, qlora_dry_run: true } });
  mockedApi.emotion.mockResolvedValue({ valence: 0.4, arousal: 0.3, optimal_loss: 1 });
  mockedApi.workingMemory.mockResolvedValue({ item_count: 2, token_count: 80, item_capacity: 10, token_capacity: 1000 });
  mockedApi.adapters.mockResolvedValue({ adapters: [{ adapter_id: "adapter-1", adapter_hash: "adapter-hash", status: "active" } as never] });
  mockedApi.contexts.mockResolvedValue({ contexts: [{ context_id: "context-1", context_type: "conversation", source_channel: "chat", source_session_id: "session-1", participant_ids: ["operator"], active_topic: null, active_task: null, status: "active", hidden_thought: "PRIVATE_SENTINEL" } as never] });
  mockedApi.goals.mockResolvedValue({ goals: [{ goal_id: "goal-1", goal_type: "intrinsic", description: "Ship safely", priority: 0.8, urgency: 0.6, confidence: 0.9, origin: "self / internal_state / endorsed", status: "active", dependency_ids: [], conflict_ids: [], deadline: null, needs_information: false, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", raw_prompt: "PRIVATE_SENTINEL" } as never], decisions: [] });
  mockedApi.commitments.mockResolvedValue({ commitments: [{ commitment_id: "commitment-1", description: "Report results", related_goal_id: "goal-1", status: "active", beneficiary: "operator", scope: "Release report", deadline: null, cost: 0.1, burden: 0.1, fulfillability: "fulfillable", fulfillability_reason: "ready", decision_refs: ["decision-1"], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", secret: "PRIVATE_SENTINEL" } as never] });
  mockedApi.plans.mockResolvedValue({ plans: [{ plan_id: "plan-1", goal_id: "goal-1", revision: 1, status: "active", steps: [{ step_id: "step-1", action_type: "respond", action_code: "report.result", dependency_ids: [], status: "in_progress", attempt_count: 1, started_at: "2026-01-01T00:00:00Z", retry_at: null, completed_at: null }], created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] });
  mockedApi.decisions.mockResolvedValue({ decisions: [{ decision_id: "decision-1", context_id: "context-1", active_goal_ids: ["goal-1"], selected_candidate_id: "candidate-1", selected_candidate: { candidate_id: "candidate-1", candidate_type: "respond", proposed_action: "Report result", plan_id: "plan-1", plan_revision: 1, step_id: "step-1", goal_refs: ["goal-1"], commitment_refs: ["commitment-1"] }, selection_confidence: 0.8, status: "awaiting_outcome", outcome_status: "pending", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }] });
  mockedApi.cockpitOutbox.mockResolvedValue({ pending_count: 42, critical_count: 9, messages: rawOutboxFixtures.map(cockpitOutboxProjection) });
  mockedApi.actionTrace.mockResolvedValue({ pending_approval_count: 2, retry_pending_count: 1, failed_count: 5, traces: rawActionFixtures.map(cockpitActionProjection) as never, pre_intent_failures: rawActionFailures.map(cockpitFailureProjection) as never });
  mockedApi.actionOperatorSummary.mockResolvedValue({ pending_approval_count: 0, operator_action_count: 0, risk_ceiling: "reversible_write", actions: [], action_tools: [], registry_tools: [] });
  mockedApi.operatorRestoreSummary.mockResolvedValue({ schema_version: 1, current_sequence: 12, current_snapshot_hash: "c".repeat(64), current_logical_digest: "d".repeat(64), semantic_revision: 4, retained_min_sequence: 1, retained_max_sequence: 12, targets: [], latest_operation: null, external_side_effects_replayed: false });
  mockedApi.cockpitTraining.mockResolvedValue(cockpitTrainingProjection());
  mockedApi.behavioralEvaluations.mockResolvedValue({ results: [{ evaluation_id: "eval-1" }] } as never);
  mockedApi.eventJournal.mockResolvedValue({ records: [{ record_id: "record-1", timestamp: "2026-01-01T00:00:00Z", lifecycle: "completed", event_id: "event-1", event_type: "goal_update", source: "runtime", processing_sequence: 1, snapshot_sequence: 1, causation_id: null, correlation_id: null, state_hash_before: null, state_hash_after: null, snapshot_hash: null, failure_category: null, actor_id: null, actor_role: null, target: "goal:goal-1", reauthenticated: null, previous_record_hash: null, record_hash: "hash", private_replay: "PRIVATE_SENTINEL" } as never] });
}

function cockpitTrainingProjection() {
  return {
    node_count: 1,
    online_node_count: 1,
    running_job_count: 1,
    failed_job_count: 0,
    importing_job_count: 0,
    active_adapter_count: 1,
    candidate_adapter_count: 0,
    nodes: [{ node_id: "node-1", role: "worker", backend: "ssh", status: "online", last_contact_at: "2026-01-01T00:00:00Z", expected_model_id: "model-1", expected_model_revision: "rev-1", expected_processor_revision: "proc-1", observed_model_id: "model-1", observed_model_revision: "rev-1", model_matches_expected: true, gpu_name: "gpu-1", cuda_version: "12.1", driver_version: "550" }],
    jobs: [{ job_id: "job-1", attempt_id: "attempt-1", status: "running", backend: "ssh", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:01Z", started_at: "2026-01-01T00:00:01Z", completed_at: null, source_event_start: 1, source_event_end: 5, selected_episode_count: 3, remote_job_id: "job-1", worker_node_id: "node-1", retry_count: 0, transferred_bytes: 2048, failure_code: null, candidate_adapter_id: "adapter-1", import_status: "not_started", bundle_digest: "a".repeat(64), result_digest: "b".repeat(64) }],
    adapters: [{ adapter_id: "adapter-1", status: "active", adapter_hash: "b".repeat(64), base_model_id: "model-1", base_model_revision: "rev-1", parent_adapter_id: null, training_job_id: "job-1", training_node_id: "node-1", submitted_by_node_id: "submitter-1", imported_by_node_id: "node-1", evaluation_id: "eval-1", evaluation_status: "passed", approved: true, active: true, rollback_candidate: false, activation_event_id: "event-1", activation_event_sequence: 1, rollback_event_id: null, rollback_event_sequence: null }],
  } as never;
}

function operatorSummary() {
  const action = {
    intent_id: "operator-action-1",
    revision: 3,
    status: "awaiting_approval" as const,
    approval: { approval_id: "approval-1", status: "pending" as const, requested_at: "2026-01-01T00:00:00Z" },
    tool: { name: "local_notification_enqueue", risk_class: "reversible_write" as const, approval_required: true, reversible: true, effect_code: "notification.enqueue", validation_schema_revision: "b".repeat(64), enabled: true, executable: true, execution_authority: "action_execution" as const },
    argument_summary: { kind: "notification" as const, channel: "local", title: "Public title", body_preview: "Public body" },
    policy: { allowed: true, approval_required: true, reason_codes: ["human_approval_required"] },
    preview: { effect_code: "notification.enqueue", effect: "Enqueue a notification", digest: "a".repeat(64), compensation_available: true },
    budget: { max_attempts: 2, max_cost_units: 1, max_monetary_cost: 0, deadline_at: "2026-01-02T00:00:00Z", attempts: 0, cost_units_used: 0, retry_at: null },
    provenance: { decision_id: "decision-1", plan_id: "plan-1", plan_revision: 1, step_id: "step-1", triggering_event_id: "event-1" },
    receipt: null,
    verification: null,
    idempotency_state: "reserved" as const,
    available_commands: ["approve", "reject", "cancel"] as Array<"approve" | "reject" | "cancel">,
    confirmation: null,
  };
  return {
    pending_approval_count: 1,
    operator_action_count: 1,
    risk_ceiling: "reversible_write" as const,
    actions: [action],
    action_tools: [action.tool],
    registry_tools: [{ name: "registry-only", description: null, tool_type: "metadata", status: "declared", generated: false, human_approved: false, execution_authority: "registry_only" as const }],
  };
}

function operatorMutationResponse(command: "approve" | "reject") {
  const action = operatorSummary().actions[0];
  return { command, event_id: "operator-event-1", processing_sequence: 11, action, disposition: command === "approve" ? "awaiting_scheduler" : "rejected" };
}

function operatorAction(overrides: Record<string, unknown> = {}) {
  return { ...operatorSummary().actions[0], ...overrides } as never;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => { resolve = resolvePromise; reject = rejectPromise; });
  return { promise, resolve, reject };
}

const rawOutboxFixtures = [
  rawOutboxFixture("message-1", "Newest message"),
  rawOutboxFixture("message-2", "Recent message 2"),
  rawOutboxFixture("message-3", "Recent message 3"),
  rawOutboxFixture("message-4", "Recent message 4"),
  rawOutboxFixture("message-5", "Recent message 5"),
  rawOutboxFixture("message-6", "Oldest message"),
];

function rawOutboxFixture(messageId: string, title: string) {
  return {
    message_id: messageId,
    urgency: "critical" as const,
    delivery_status: "pending" as const,
    acknowledgment_status: "unacknowledged" as const,
    title,
    references: { decision_id: "decision-1", action_id: "action-1", event_id: null, goal_id: "goal-1", plan_id: "plan-1", commitment_id: "commitment-1" },
    body: "PRIVATE_SENTINEL",
    responses: [{ text: "PRIVATE_SENTINEL" }],
    attempts: [{ failure_code: "PRIVATE_SENTINEL" }],
  };
}

function cockpitOutboxProjection(message: ReturnType<typeof rawOutboxFixture>) {
  return {
    message_id: message.message_id,
    urgency: message.urgency,
    delivery_status: message.delivery_status,
    acknowledgment_status: message.acknowledgment_status,
    title: message.title,
    references: message.references,
  };
}

const rawActionFixtures = [
  rawActionFixture("action-1", "succeeded", "receipt-1", "succeeded"),
  { ...rawActionFixture("action-old", "succeeded", null, null), revision: 99, updated_at: "2026-07-29T00:00:00Z", provenance: { decision_id: "decision-1", candidate_id: "candidate-1", triggering_event_id: "event-old", plan_id: "plan-1", plan_revision: 1, step_id: "step-1" } },
  rawActionFixture("action-pending", "awaiting_approval", null, null),
  rawActionFixture("action-retry", "retry_pending", null, null),
  rawActionFixture("action-failed", "failed", "receipt-failed", "timed_out"),
  rawActionFixture("action-compensated", "compensated", "receipt-compensated", "compensated"),
];

function rawActionFixture(intentId: string, status: string, receiptId: string | null, receiptStatus: string | null) {
  return {
    intent_id: intentId,
    revision: 2,
    tool_name: "document_search",
    risk_class: "read_only",
    status,
    dry_run: false,
    created_at: "2026-07-30T00:00:00Z",
    updated_at: "2026-07-30T00:00:01Z",
    failure_code: status === "failed" || status === "retry_pending" ? "timeout" : null,
    provenance: { decision_id: intentId === "action-1" ? "decision-1" : `decision-${intentId}`, candidate_id: "candidate-1", triggering_event_id: "event-0", plan_id: "plan-1", plan_revision: 1, step_id: "step-1" },
    approval: { approval_id: status === "awaiting_approval" ? "approval-1" : null, status: status === "awaiting_approval" ? "pending" : null, requested_at: status === "awaiting_approval" ? "2026-07-30T00:00:00Z" : null, resolved_at: null, resolved_by_operator: false, reason: "PRIVATE_SENTINEL" },
    receipt: receiptId && receiptStatus ? { receipt_id: receiptId, status: receiptStatus, attempt: 1, duration_ms: 12.5, event_id: intentId === "action-1" ? "event-1" : null, event_sequence: intentId === "action-1" ? 45 : null, error_code: status === "failed" ? "timeout" : null, compensation_of: status === "compensated" ? "receipt-original" : null, idempotency_key: "PRIVATE_SENTINEL" } : null,
    related_receipts: status === "compensated" ? [{ receipt_id: "receipt-original", status: "succeeded", idempotency_key: "PRIVATE_SENTINEL" }] : [],
    observation: intentId === "action-1" ? { observation_id: "observation-1", valid: true, validation_errors: [], result_digest: "a".repeat(64), data: { result: "PRIVATE_SENTINEL" } } : null,
    verification: intentId === "action-1" ? { verification_id: "verification-1", success: true, reason: "observation_schema_valid", private_replay: "PRIVATE_SENTINEL" } : null,
    arguments: { query: "PRIVATE_SENTINEL" },
    preview: { arguments: { query: "PRIVATE_SENTINEL" } },
    idempotency_key: "PRIVATE_SENTINEL",
  };
}

function cockpitActionProjection(action: ReturnType<typeof rawActionFixture>) {
  return {
    intent_id: action.intent_id,
    revision: action.revision,
    tool_name: action.tool_name,
    risk_class: action.risk_class,
    status: action.status,
    dry_run: action.dry_run,
    created_at: action.created_at,
    updated_at: action.updated_at,
    failure_code: action.failure_code,
    provenance: action.provenance,
    approval: {
      approval_id: action.approval.approval_id,
      status: action.approval.status,
      requested_at: action.approval.requested_at,
      resolved_at: action.approval.resolved_at,
      resolved_by_operator: action.approval.resolved_by_operator,
    },
    receipt: action.receipt && {
      receipt_id: action.receipt.receipt_id,
      status: action.receipt.status,
      attempt: action.receipt.attempt,
      duration_ms: action.receipt.duration_ms,
      event_id: action.receipt.event_id,
      event_sequence: action.receipt.event_sequence,
      error_code: action.receipt.error_code,
      compensation_of: action.receipt.compensation_of,
    },
    related_receipts: action.related_receipts.map((receipt) => ({ receipt_id: receipt.receipt_id, status: receipt.status })),
    observation: action.observation && {
      observation_id: action.observation.observation_id,
      valid: action.observation.valid,
      validation_errors: action.observation.validation_errors,
      result_digest: action.observation.result_digest,
    },
    verification: action.verification && {
      verification_id: action.verification.verification_id,
      success: action.verification.success,
      reason: action.verification.reason,
    },
  };
}

const rawActionFailures = [
  { failure_id: "validation-1", failure_type: "validation", decision_id: "decision-1", candidate_id: null, tool_name: "document_search", risk_class: "read_only", error_codes: ["arguments_schema_invalid"], event_id: "event-1", event_sequence: 42, occurred_at: "2026-07-30T00:00:02Z", idempotency_key: "PRIVATE_SENTINEL", request_digest: "PRIVATE_SENTINEL", canonical_arguments_digest: "PRIVATE_SENTINEL", arguments: { secret: "PRIVATE_SENTINEL" } },
  { failure_id: "rejection-1", failure_type: "policy_rejection", decision_id: "decision-2", candidate_id: "candidate-2", tool_name: "PRIVATE_SENTINEL<script>", risk_class: "reversible_write", error_codes: ["risk_class_exceeds_budget"], event_id: "event-2", event_sequence: 43, occurred_at: "2026-07-30T00:00:01Z", idempotency_key: "PRIVATE_SENTINEL" },
];

function cockpitFailureProjection(failure: typeof rawActionFailures[number]) {
  return {
    failure_id: failure.failure_id,
    failure_type: failure.failure_type,
    decision_id: failure.decision_id,
    candidate_id: failure.candidate_id,
    tool_name: failure.tool_name === "document_search" ? failure.tool_name : null,
    risk_class: failure.risk_class,
    error_codes: failure.error_codes,
    event_id: failure.event_id,
    event_sequence: failure.event_sequence,
    occurred_at: failure.occurred_at,
  };
}
