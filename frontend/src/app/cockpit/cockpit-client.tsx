"use client";

import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import Link from "next/link";
import { useRef, useState, type ReactNode } from "react";
import { EmotionMeter } from "@/components/emotion-meter";
import { Badge } from "@/components/ui/badge";
import { Card, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  api,
  errorMessage,
  type CockpitActionTrace,
  type CockpitAdapterLineage,
  type CockpitPreIntentFailure,
  type CockpitTrainingJob,
  type CockpitTrainingNode,
  type Commitment,
  type ContextFrame,
  type Decision,
  type Goal,
  type JournalRecord,
  type CockpitOutboxMessage,
  type Plan,
  type OperatorAction,
  type ActionTool,
  type RegistryTool,
} from "@/lib/api";
import { evaluationHref } from "@/lib/anchors";
import { actionMutationInvalidationKeys, queryKeys } from "@/lib/query-keys";

type OperatorCommand = OperatorAction["available_commands"][number];
type OperatorMutationVariables = { command: OperatorCommand; action: OperatorAction; reason?: string; phrase?: string };

export function CockpitClient() {
  const queryClient = useQueryClient();
  const runtime = useQuery({ queryKey: queryKeys.cockpit.runtime, queryFn: api.systemInfo });
  const emotion = useQuery({ queryKey: queryKeys.cockpit.emotion, queryFn: api.emotion });
  const workingMemory = useQuery({ queryKey: queryKeys.cockpit.workingMemory, queryFn: api.workingMemory });
  const contexts = useQuery({ queryKey: queryKeys.cockpit.contexts, queryFn: api.contexts });
  const goals = useQuery({ queryKey: queryKeys.cockpit.goals, queryFn: api.goals });
  const commitments = useQuery({ queryKey: queryKeys.cockpit.commitments, queryFn: api.commitments });
  const plans = useQuery({ queryKey: queryKeys.cockpit.plans, queryFn: api.plans });
  const decisions = useQuery({ queryKey: queryKeys.cockpit.decisions, queryFn: api.decisions });
  const outbox = useQuery({ queryKey: queryKeys.cockpit.outbox, queryFn: api.cockpitOutbox });
  const actions = useQuery({ queryKey: queryKeys.cockpit.actions, queryFn: api.actionTrace });
  const training = useQuery({ queryKey: queryKeys.cockpit.training, queryFn: api.cockpitTraining });
  const evaluations = useQuery({ queryKey: queryKeys.cockpit.evaluations, queryFn: api.behavioralEvaluations });
  const journal = useQuery({ queryKey: queryKeys.cockpit.journal, queryFn: api.eventJournal });
  const adapters = useQuery({ queryKey: queryKeys.cockpit.adapters, queryFn: api.adapters });
  const pendingIntents = useRef(new Set<string>());
  const [pendingIntentIds, setPendingIntentIds] = useState<Set<string>>(new Set());
  const operator = useQuery({
    queryKey: queryKeys.cockpit.actionOperator,
    queryFn: api.actionOperatorSummary,
    refetchInterval: (query) => hasActiveOperatorItems(query.state.data) ? 5000 : false,
  });
  const operatorMutation = useMutation({
    mutationFn: ({ command, action, reason, phrase }: OperatorMutationVariables) => {
      const common = {
        expected_intent_revision: action.revision,
        expected_preview_digest: action.preview.digest,
        ...(phrase ? { confirmation_phrase: phrase } : {}),
      };
      if (command === "approve" || command === "reject") {
        if (action.approval === null) throw new Error("Approval binding is unavailable");
        const request = {
          ...common,
          expected_approval_id: action.approval.approval_id,
          ...(reason ? { reason } : {}),
        };
        return command === "approve"
          ? api.approveAction(action.intent_id, request)
          : api.rejectAction(action.intent_id, request);
      }
      if (command === "cancel") return api.cancelAction(action.intent_id, common);
      if (command === "retry_now") return api.retryAction(action.intent_id, common);
      return api.compensateAction(action.intent_id, common);
    },
    onSettled: async (_data, _error, variables) => {
      try {
        await Promise.all(actionMutationInvalidationKeys.map((queryKey) => queryClient.invalidateQueries({ queryKey, refetchType: "all" })));
      } finally {
        if (variables) {
          pendingIntents.current.delete(variables.action.intent_id);
          setPendingIntentIds(new Set(pendingIntents.current));
        }
      }
    },
  });

  const currentGoals = goals.data?.goals.filter(isCurrentGoal) ?? [];
  const currentCommitments = commitments.data?.commitments.filter(isCurrentCommitment) ?? [];
  const activePlans = plans.data?.plans.filter((plan) => plan.status === "active") ?? [];
  const recentDecisions = decisions.data?.decisions.slice(-6).reverse() ?? [];
  const recentOutbox = outbox.data?.messages.slice(0, 5) ?? [];
  const recentJournal = journal.data?.records.slice(-6).reverse() ?? [];
  const loadedContexts = new Set(contexts.data?.contexts.map((item) => item.context_id));
  const loadedGoals = new Set(currentGoals.map((item) => item.goal_id));
  const loadedCommitments = new Set(currentCommitments.map((item) => item.commitment_id));
  const loadedPlans = new Set(activePlans.map((item) => item.plan_id));
  const loadedDecisions = new Set(recentDecisions.map((item) => item.decision_id));
  const loadedOutbox = new Set(recentOutbox.map((item) => item.message_id));
  const loadedJournal = new Set(recentJournal.map((item) => item.event_id));
  const loadedActions = new Set(actions.data?.traces.map((item) => item.intent_id));
  const loadedReceipts = new Set(actions.data?.traces.flatMap((item) => [
    ...(item.receipt ? [item.receipt.receipt_id] : []),
    ...item.related_receipts.map((receipt) => receipt.receipt_id),
  ]));
  const loadedActionFailures = new Set(actions.data?.pre_intent_failures.map((item) => item.failure_id));
  const loadedTrainingNodes = new Set(training.data?.nodes.map((item) => item.node_id));
  const loadedTrainingJobs = new Set(training.data?.jobs.map((item) => item.job_id));
  const loadedCockpitAdapters = new Set(training.data?.adapters.map((item) => item.adapter_id));
  const loadedEvaluations = new Set(evaluations.data?.results.map((item) => item.evaluation_id));
  const activeAdapter = adapters.data?.adapters.find((adapter) => adapter.status === "active");
  const operatorActions = operator.data?.actions ?? [];
  const approvalInbox = operatorActions.filter((item) => item.status === "awaiting_approval");

  return (
    <div className="page cockpit-page">
      <header className="page-header">
        <div>
          <p className="cockpit-kicker">SUBJECT / GOVERNED OPERATOR</p>
          <h1 className="page-title">Cockpit</h1>
          <p className="page-subtitle">Current runtime state and traceable Goal → Plan → Decision → Action / Outbox references.</p>
        </div>
        <div className="metadata-row"><Link className="entity-link" href="/decisions">Decision explanations</Link><Link className="entity-link" href="/outbox">Open outbox</Link></div>
      </header>

      <div className="cockpit-grid">
        <Section title="Runtime" query={runtime} empty={false}>
          {runtime.data ? <div className="metric-grid"><Metric label="Project" value={runtime.data.project} /><Metric label="Status" value={runtime.data.status} /><Metric label="Version" value={runtime.data.build.version} /><Metric label="Environment" value={runtime.data.runtime.environment} /><Metric label="Provider" value={runtime.data.runtime.provider} /><Metric label="Model" value={runtime.data.runtime.primary_model_id} /></div> : null}
          {runtime.data?.build.commit ? <p className="mono muted">build {runtime.data.build.commit}</p> : null}
          <QueryState query={adapters} loading="Loading active adapter..." empty={false} />
          {adapters.data ? <p>Active adapter: {activeAdapter ? <span className="mono">{activeAdapter.adapter_id} / {activeAdapter.adapter_hash ?? "hash unavailable"}</span> : "none"}</p> : null}
        </Section>

        <Card aria-labelledby="emotion-memory-title">
          <CardTitle id="emotion-memory-title">Emotion / Working Memory</CardTitle>
          <QueryState query={emotion} loading="Loading emotion..." empty={false} />
          {emotion.data ? <EmotionMeter emotion={emotion.data} /> : null}
          <QueryState query={workingMemory} loading="Loading working memory..." empty={false} />
          {workingMemory.data ? <div className="capacity-grid"><Capacity label="Items" value={workingMemory.data.item_count} capacity={workingMemory.data.item_capacity} /><Capacity label="Tokens" value={workingMemory.data.token_count} capacity={workingMemory.data.token_capacity} /></div> : null}
        </Card>

        <Section title="Contexts" query={contexts} empty={contexts.data?.contexts.length === 0} emptyText="No contexts recorded.">
          <div className="stack">{contexts.data?.contexts.map((context) => <ContextRecord key={context.context_id} context={context} decisions={decisions.data?.decisions ?? []} loadedDecisions={loadedDecisions} />)}</div>
        </Section>

        <Card aria-labelledby="goals-commitments-title">
          <CardTitle id="goals-commitments-title">Goals / Commitments</CardTitle>
          <h3 className="cockpit-subheading">Active and candidate goals</h3>
          <QueryState query={goals} loading="Loading goals..." empty={currentGoals.length === 0} emptyText="No active or candidate goals." />
          <div className="stack">{currentGoals.map((goal) => <GoalRecord key={goal.goal_id} goal={goal} plans={plans.data?.plans ?? []} decisions={decisions.data?.decisions ?? []} loadedGoals={loadedGoals} loadedPlans={loadedPlans} loadedDecisions={loadedDecisions} />)}</div>
          <h3 className="cockpit-subheading">Current commitments</h3>
          <QueryState query={commitments} loading="Loading commitments..." empty={currentCommitments.length === 0} emptyText="No current commitments." />
          <div className="stack">{currentCommitments.map((commitment) => <CommitmentRecord key={commitment.commitment_id} commitment={commitment} loadedGoals={loadedGoals} loadedDecisions={loadedDecisions} />)}</div>
        </Card>

        <Section title="Plans / Steps" query={plans} empty={activePlans.length === 0} emptyText="No active plans.">
          <div className="stack">{activePlans.map((plan) => <PlanRecord key={plan.plan_id} plan={plan} decisions={decisions.data?.decisions ?? []} loadedGoals={loadedGoals} loadedDecisions={loadedDecisions} />)}</div>
        </Section>

        <Section title="Recent Decisions" query={decisions} empty={decisions.data?.decisions.length === 0} emptyText="No decisions recorded.">
          <div className="stack">{recentDecisions.map((decision) => <DecisionRecord key={decision.decision_id} decision={decision} actions={actions.data?.traces ?? []} actionFailures={actions.data?.pre_intent_failures ?? []} outbox={outbox.data?.messages ?? []} loadedContexts={loadedContexts} loadedGoals={loadedGoals} loadedPlans={loadedPlans} loadedCommitments={loadedCommitments} loadedOutbox={loadedOutbox} loadedActions={loadedActions} loadedActionFailures={loadedActionFailures} />)}</div>
        </Section>

         <Section title="Action Execution" query={actions} empty={actions.data ? actions.data.traces.length === 0 && actions.data.pre_intent_failures.length === 0 : false} emptyText="No action execution traces.">
          {actions.data ? <div className="metric-grid"><Metric label="Pending approvals" value={String(actions.data.pending_approval_count)} /><Metric label="Retry pending" value={String(actions.data.retry_pending_count)} /><Metric label="Failed" value={String(actions.data.failed_count)} /></div> : null}
          <div className="stack">{actions.data?.pre_intent_failures.map((failure) => <PreIntentFailureRecord key={failure.failure_id} failure={failure} loadedDecisions={loadedDecisions} loadedJournal={loadedJournal} />)}</div>
          <div className="stack">{actions.data?.traces.map((trace) => <ActionTraceRecord key={trace.intent_id} trace={trace} loadedDecisions={loadedDecisions} loadedPlans={loadedPlans} loadedJournal={loadedJournal} loadedReceipts={loadedReceipts} />)}</div>
         </Section>

         <Section title="Action Operator" query={operator} empty={operator.data ? operatorActions.length === 0 : false} emptyText="No operator actions.">
           {operator.data ? <ActionOperator actions={operatorActions} approvalInbox={approvalInbox} actionTools={operator.data.action_tools} registryTools={operator.data.registry_tools} pendingIntentIds={pendingIntentIds} onRun={(action, command, reason, phrase) => { if (pendingIntents.current.has(action.intent_id)) return; pendingIntents.current.add(action.intent_id); setPendingIntentIds(new Set(pendingIntents.current)); operatorMutation.mutate({ command, action, reason, phrase }); }} loadedDecisions={loadedDecisions} loadedPlans={loadedPlans} loadedJournal={loadedJournal} loadedReceipts={loadedReceipts} /> : null}
         </Section>

        <Section title="Training / Adapters" query={training} empty={training.data ? training.data.nodes.length === 0 && training.data.jobs.length === 0 && training.data.adapters.length === 0 : false} emptyText="No training nodes, jobs, or adapter lineage records.">
          {training.data ? <TrainingAdaptersSummary nodes={training.data.node_count} online={training.data.online_node_count} running={training.data.running_job_count} failed={training.data.failed_job_count} importing={training.data.importing_job_count} activeAdapters={training.data.active_adapter_count} candidateAdapters={training.data.candidate_adapter_count} /> : null}
          <div className="stack">{training.data?.nodes.map((node) => <TrainingNodeRecord key={node.node_id} node={node} jobs={training.data?.jobs ?? []} loadedJobs={loadedTrainingJobs} />)}</div>
          <div className="stack">{training.data?.jobs.map((job) => <TrainingJobRecord key={job.job_id} job={job} loadedNodes={loadedTrainingNodes} loadedAdapters={loadedCockpitAdapters} />)}</div>
          <div className="stack">{training.data?.adapters.map((adapter) => <AdapterLineageRecord key={adapter.adapter_id} adapter={adapter} loadedJobs={loadedTrainingJobs} loadedNodes={loadedTrainingNodes} loadedEvaluations={loadedEvaluations} loadedJournal={loadedJournal} />)}</div>
        </Section>

        <Section title="Outbox" query={outbox} empty={outbox.data?.messages.length === 0} emptyText="No proactive messages.">
          {outbox.data ? <OutboxSummary pendingCount={outbox.data.pending_count} criticalCount={outbox.data.critical_count} visibleMessages={recentOutbox} loadedGoals={loadedGoals} loadedPlans={loadedPlans} loadedDecisions={loadedDecisions} loadedCommitments={loadedCommitments} loadedActions={loadedActions} /> : null}
        </Section>

        <Section title="Recent Journal" query={journal} empty={journal.data?.records.length === 0} emptyText="No Journal records.">
          <div className="stack">{recentJournal.map((record) => <JournalEntry key={record.record_id} record={record} actions={actions.data?.traces ?? []} loadedJournal={loadedJournal} loadedActions={loadedActions} loadedReceipts={loadedReceipts} />)}</div>
        </Section>
      </div>
    </div>
  );
}

function Section<T>({ title, query, empty, emptyText, children }: { title: string; query: UseQueryResult<T>; empty: boolean; emptyText?: string; children: ReactNode }) {
  return <Card><CardTitle>{title}</CardTitle><QueryState query={query} loading={`Loading ${title.toLowerCase()}...`} empty={empty} emptyText={emptyText} />{children}</Card>;
}

function QueryState<T>({ query, loading, empty, emptyText }: { query: UseQueryResult<T>; loading: string; empty: boolean; emptyText?: string }) {
  if (query.isPending) return <p className="muted">{loading}</p>;
  if (query.error) return <p className="error">{errorMessage(query.error)}</p>;
  if (empty) return <p className="muted">{emptyText ?? "No records."}</p>;
  return null;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric-cell"><span className="muted">{label}</span><strong>{value}</strong></div>;
}

function Capacity({ label, value, capacity }: { label: string; value: number; capacity: number }) {
  const percentage = capacity ? Math.min(100, Math.round(value / capacity * 100)) : 0;
  return <div><div className="capacity-label"><span>{label}</span><span>{value} / {capacity}</span></div><div className="capacity-track"><span style={{ width: `${percentage}%` }} /></div></div>;
}

function ContextRecord({ context, decisions, loadedDecisions }: { context: ContextFrame; decisions: Decision[]; loadedDecisions: Set<string> }) {
  const related = decisions.filter((decision) => decision.context_id === context.context_id);
  return <article className="record" id={anchor("context", context.context_id)}><div className="metadata-row"><Badge>{context.status}</Badge><strong>{context.context_type}</strong></div><p className="mono muted">{context.context_id} · {context.source_channel}{context.source_session_id ? ` / ${context.source_session_id}` : ""}</p><p>Interlocutors: {context.participant_ids.join(", ") || "none"}</p><p>Topic: {context.active_topic ?? "none"} · Task: {context.active_task ?? "none"}</p><p>Decisions: {related.length ? <ReferenceList kind="decision" ids={related.map((item) => item.decision_id)} available={loadedDecisions} /> : "unavailable"}</p></article>;
}

function GoalRecord({ goal, plans, decisions, loadedGoals, loadedPlans, loadedDecisions }: { goal: Goal; plans: Plan[]; decisions: Decision[]; loadedGoals: Set<string>; loadedPlans: Set<string>; loadedDecisions: Set<string> }) {
  const goalPlans = plans.filter((plan) => plan.goal_id === goal.goal_id).map((plan) => plan.plan_id);
  const goalDecisions = decisions.filter((decision) => decision.active_goal_ids.includes(goal.goal_id) || decision.selected_candidate.goal_refs.includes(goal.goal_id)).map((decision) => decision.decision_id);
  return <article className="record" id={anchor("goal", goal.goal_id)}><div className="metadata-row"><Badge>{goal.status}</Badge><Badge>{goal.goal_type}</Badge><strong>{goal.description}</strong></div><p className="muted">Priority {percent(goal.priority)} · urgency {percent(goal.urgency)} · confidence {percent(goal.confidence)} · origin {goal.origin}</p><p>Dependencies: {goal.dependency_ids.length ? <ReferenceList kind="goal" ids={goal.dependency_ids} available={loadedGoals} /> : "none"} · Conflicts: {goal.conflict_ids.length ? <ReferenceList kind="goal" ids={goal.conflict_ids} available={loadedGoals} /> : "none"}</p><p>Plans: {goalPlans.length ? <ReferenceList kind="plan" ids={goalPlans} available={loadedPlans} /> : "unavailable"} · Decisions: {goalDecisions.length ? <ReferenceList kind="decision" ids={goalDecisions} available={loadedDecisions} /> : "unavailable"}</p></article>;
}

function CommitmentRecord({ commitment, loadedGoals, loadedDecisions }: { commitment: Commitment; loadedGoals: Set<string>; loadedDecisions: Set<string> }) {
  return <article className="record" id={anchor("commitment", commitment.commitment_id)}><div className="metadata-row"><Badge>{commitment.status}</Badge><Badge data-tone={commitment.fulfillability === "at_risk" || commitment.fulfillability === "impossible" ? "danger" : "neutral"}>{commitment.fulfillability}</Badge><strong>{commitment.description}</strong></div><p>{commitment.scope} · beneficiary {commitment.beneficiary} · deadline {commitment.deadline ?? "none"}</p><p>Goal: <EntityReference kind="goal" id={commitment.related_goal_id} available={loadedGoals} /> · Decisions: {commitment.decision_refs.length ? <ReferenceList kind="decision" ids={commitment.decision_refs} available={loadedDecisions} /> : "unavailable"}</p></article>;
}

function PlanRecord({ plan, decisions, loadedGoals, loadedDecisions }: { plan: Plan; decisions: Decision[]; loadedGoals: Set<string>; loadedDecisions: Set<string> }) {
  const currentStepId = plan.steps.find((step) => step.status === "in_progress")?.step_id
    ?? plan.steps.find((step) => step.status === "ready" || step.status === "waiting_retry")?.step_id
    ?? plan.steps.find((step) => step.status === "pending")?.step_id;
  return <article className="record" id={anchor("plan", plan.plan_id)}><div className="metadata-row"><Badge data-tone="accent">{plan.status}</Badge><strong>{plan.plan_id}</strong><span>revision {plan.revision}</span></div><p>Goal: <EntityReference kind="goal" id={plan.goal_id} available={loadedGoals} /></p><div className="step-list">{plan.steps.map((step) => {
    const related = decisions.filter((decision) => decision.selected_candidate.plan_id === plan.plan_id && decision.selected_candidate.step_id === step.step_id).map((decision) => decision.decision_id);
    return <div className="step-row" id={anchor("step", `${plan.plan_id}-${step.step_id}`)} key={step.step_id}><span className="metadata-row"><Badge data-tone={step.status === "failed" ? "danger" : step.status === "in_progress" || step.status === "ready" ? "accent" : "neutral"}>{step.status}</Badge>{step.step_id === currentStepId ? <Badge data-tone="accent">current</Badge> : null}</span><span><strong>{step.action_code}</strong><span className="muted"> · {step.action_type} · attempt {step.attempt_count}</span></span><span>Decision: {related.length ? <ReferenceList kind="decision" ids={related} available={loadedDecisions} /> : "unavailable"}</span></div>;
  })}</div></article>;
}

function DecisionRecord({ decision, actions, actionFailures, outbox, loadedContexts, loadedGoals, loadedPlans, loadedCommitments, loadedOutbox, loadedActions, loadedActionFailures }: { decision: Decision; actions: CockpitActionTrace[]; actionFailures: CockpitPreIntentFailure[]; outbox: CockpitOutboxMessage[]; loadedContexts: Set<string>; loadedGoals: Set<string>; loadedPlans: Set<string>; loadedCommitments: Set<string>; loadedOutbox: Set<string>; loadedActions: Set<string>; loadedActionFailures: Set<string> }) {
  const messages = outbox.filter((message) => message.references.decision_id === decision.decision_id).map((message) => message.message_id);
  const candidate = decision.selected_candidate;
  const action = actions.find((trace) => trace.provenance.decision_id === decision.decision_id && trace.provenance.candidate_id === decision.selected_candidate_id);
  const failure = actionFailures.find((item) => item.decision_id === decision.decision_id && (item.candidate_id === null || item.candidate_id === decision.selected_candidate_id));
  const actionReference = action
    ? <EntityReference kind="action" id={action.intent_id} available={loadedActions} />
    : <EntityReference kind="action-failure" id={failure?.failure_id ?? null} available={loadedActionFailures} />;
  return <article className="record" id={anchor("decision", decision.decision_id)}><div className="metadata-row"><Badge>{decision.status}</Badge><Badge>{decision.outcome_status}</Badge><strong>{decision.decision_id}</strong><span>{percent(decision.selection_confidence)} confidence</span></div><p>Selected action: <span className="mono">{candidate.candidate_type}:{candidate.candidate_id}</span> · {candidate.proposed_action}</p><p>Context: <EntityReference kind="context" id={decision.context_id} available={loadedContexts} /> · Plan: <EntityReference kind="plan" id={candidate.plan_id} available={loadedPlans} />{candidate.step_id ? ` / step ${candidate.step_id}` : ""}</p><p>Goals: {decision.active_goal_ids.length ? <ReferenceList kind="goal" ids={decision.active_goal_ids} available={loadedGoals} /> : "unavailable"} · Commitments: {candidate.commitment_refs.length ? <ReferenceList kind="commitment" ids={candidate.commitment_refs} available={loadedCommitments} /> : "unavailable"}</p><p>Action: {actionReference} · Outbox: {messages.length ? <ReferenceList kind="outbox" ids={messages} available={loadedOutbox} /> : "unavailable"}</p></article>;
}

function PreIntentFailureRecord({ failure, loadedDecisions, loadedJournal }: { failure: CockpitPreIntentFailure; loadedDecisions: Set<string>; loadedJournal: Set<string> }) {
  return <article className="record" id={anchor("action-failure", failure.failure_id)}><div className="metadata-row"><Badge data-tone="danger">{failure.failure_type === "validation" ? "Validation rejected" : "Policy rejected"}</Badge><strong>{failure.failure_id}</strong></div><p>Decision: <EntityReference kind="decision" id={failure.decision_id} available={loadedDecisions} /> · Candidate: <span className="mono">{failure.candidate_id ?? "unavailable"}</span></p><p>Tool: {failure.tool_name ?? "unavailable"} · Risk: {failure.risk_class ?? "unavailable"} · Codes: <span className="mono">{failure.error_codes.join(", ")}</span></p><p>Event: <EntityReference kind="journal" id={failure.event_id} available={loadedJournal} /> · {failure.occurred_at}</p></article>;
}

function ActionTraceRecord({ trace, loadedDecisions, loadedPlans, loadedJournal, loadedReceipts }: { trace: CockpitActionTrace; loadedDecisions: Set<string>; loadedPlans: Set<string>; loadedJournal: Set<string>; loadedReceipts: Set<string> }) {
  const receipt = trace.receipt;
  return <article className="record" id={anchor("action", trace.intent_id)}><div className="metadata-row"><Badge data-tone={actionTone(trace.status)}>{trace.status}</Badge><Badge>{trace.risk_class}</Badge><strong>{trace.tool_name}</strong></div><p className="mono muted">{trace.intent_id} · revision {trace.revision} · {trace.dry_run ? "dry-run" : "live"}</p><p>Created: {trace.created_at} · Updated: {trace.updated_at} · Failure: <span className="mono">{trace.failure_code ?? "none"}</span></p><p>Decision: <EntityReference kind="decision" id={trace.provenance.decision_id} available={loadedDecisions} /> · Plan: <EntityReference kind="plan" id={trace.provenance.plan_id} available={loadedPlans} />{trace.provenance.step_id ? ` / step ${trace.provenance.step_id}` : ""}</p><p>Approval: {trace.approval.status ?? "not required"}{trace.approval.resolved_by_operator ? " / operator resolved" : ""}</p>{trace.related_receipts.map((related) => <p id={anchor("receipt", related.receipt_id)} key={related.receipt_id}>Related receipt: <Link className="entity-link mono" href={`#${anchor("receipt", related.receipt_id)}`}>{related.receipt_id}</Link> · {related.status}</p>)}{receipt ? <div className="step-row" id={anchor("receipt", receipt.receipt_id)}><p>Receipt: <Link className="entity-link mono" href={`#${anchor("receipt", receipt.receipt_id)}`}>{receipt.receipt_id}</Link> · <Badge data-tone={receipt.status === "succeeded" ? "success" : receipt.status === "failed" || receipt.status === "timed_out" || receipt.status === "cancelled" ? "danger" : "neutral"}>{receipt.status}</Badge></p>{receipt.compensation_of ? <p>Compensates: <EntityReference kind="receipt" id={receipt.compensation_of} available={loadedReceipts} /></p> : null}<p>Attempt {receipt.attempt} · {receipt.duration_ms} ms · Error {receipt.error_code ?? "none"} · Event <EntityReference kind="journal" id={receipt.event_id} available={loadedJournal} /></p>{trace.observation ? <p id={anchor("observation", trace.observation.observation_id)}>Observation: <Link className="entity-link mono" href={`#${anchor("observation", trace.observation.observation_id)}`}>{trace.observation.observation_id}</Link> · {trace.observation.valid ? "valid" : "invalid"} · errors {trace.observation.validation_errors.join(", ") || "none"} · digest <span className="mono">{trace.observation.result_digest}</span></p> : <p>Observation: unavailable</p>}{trace.verification ? <p id={anchor("verification", trace.verification.verification_id)}>Verification: <Link className="entity-link mono" href={`#${anchor("verification", trace.verification.verification_id)}`}>{trace.verification.verification_id}</Link> · {trace.verification.success ? "succeeded" : "failed"} · {trace.verification.reason}</p> : <p>Verification: unavailable</p>}</div> : <p>Receipt: unavailable · Observation: unavailable · Verification: unavailable</p>}</article>;
}

function TrainingAdaptersSummary({ nodes, online, running, failed, importing, activeAdapters, candidateAdapters }: { nodes: number; online: number; running: number; failed: number; importing: number; activeAdapters: number; candidateAdapters: number }) {
  return <div className="metric-grid"><Metric label="Nodes" value={String(nodes)} /><Metric label="Online" value={String(online)} /><Metric label="Running" value={String(running)} /><Metric label="Failed" value={String(failed)} /><Metric label="Importing" value={String(importing)} /><Metric label="Active adapters" value={String(activeAdapters)} /><Metric label="Candidate adapters" value={String(candidateAdapters)} /></div>;
}

function ActionOperator({ actions, approvalInbox, actionTools, registryTools, pendingIntentIds, onRun, loadedDecisions, loadedPlans, loadedJournal, loadedReceipts }: { actions: OperatorAction[]; approvalInbox: OperatorAction[]; actionTools: ActionTool[]; registryTools: RegistryTool[]; pendingIntentIds: Set<string>; onRun: (action: OperatorAction, command: OperatorCommand, reason?: string, phrase?: string) => void; loadedDecisions: Set<string>; loadedPlans: Set<string>; loadedJournal: Set<string>; loadedReceipts: Set<string> }) {
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [phrases, setPhrases] = useState<Record<string, string>>({});
  const approvalIds = new Set(approvalInbox.map((action) => action.intent_id));
  const run = (action: OperatorAction, command: OperatorCommand) => onRun(action, command, reasons[action.intent_id], phrases[action.intent_id]);
  return <div className="stack">
    <h3 className="cockpit-subheading">Approval Inbox</h3>
    {approvalInbox.length === 0 ? <p className="muted">No pending approvals.</p> : approvalInbox.map((action) => <OperatorRecord key={action.intent_id} action={action} pending={pendingIntentIds.has(action.intent_id)} reason={reasons[action.intent_id] ?? ""} phrase={phrases[action.intent_id] ?? ""} onReason={(value) => setReasons((current) => ({ ...current, [action.intent_id]: value.slice(0, 500) }))} onPhrase={(value) => setPhrases((current) => ({ ...current, [action.intent_id]: value }))} onRun={run} loadedDecisions={loadedDecisions} loadedPlans={loadedPlans} loadedJournal={loadedJournal} loadedReceipts={loadedReceipts} />)}
    <h3 className="cockpit-subheading">Other operator actions</h3>
    {actions.filter((action) => !approvalIds.has(action.intent_id)).map((action) => <OperatorRecord key={action.intent_id} action={action} pending={pendingIntentIds.has(action.intent_id)} reason={reasons[action.intent_id] ?? ""} phrase={phrases[action.intent_id] ?? ""} onReason={(value) => setReasons((current) => ({ ...current, [action.intent_id]: value.slice(0, 500) }))} onPhrase={(value) => setPhrases((current) => ({ ...current, [action.intent_id]: value }))} onRun={run} loadedDecisions={loadedDecisions} loadedPlans={loadedPlans} loadedJournal={loadedJournal} loadedReceipts={loadedReceipts} />)}
    <h3 className="cockpit-subheading">Action-executable tools</h3>
    <ActionToolCatalog tools={actionTools} />
    <h3 className="cockpit-subheading">Registry-only tools</h3>
    <RegistryToolCatalog tools={registryTools} />
  </div>;
}

function OperatorRecord({ action, pending, reason, phrase, onReason, onPhrase, onRun, loadedDecisions, loadedPlans, loadedJournal, loadedReceipts }: { action: OperatorAction; pending: boolean; reason: string; phrase: string; onReason: (value: string) => void; onPhrase: (value: string) => void; onRun: (action: OperatorAction, command: OperatorCommand) => void; loadedDecisions: Set<string>; loadedPlans: Set<string>; loadedJournal: Set<string>; loadedReceipts: Set<string> }) {
  const commands = action.available_commands;
  const confirmationNeeded = action.confirmation?.required === true;
  return <article className="record" id={anchor("action", action.intent_id)}>
    <div className="metadata-row"><Badge>{action.status}</Badge><Badge>{action.tool.risk_class}</Badge><strong>{action.tool.name}</strong><span className="mono">{action.intent_id}</span></div>
    <p>{formatArgumentSummary(action)}</p>
    <p>Effect: {action.preview.effect} · Policy: {action.policy.reason_codes.join(", ")}</p>
    <p>Preview digest: <span className="mono">{action.preview.digest}</span> · Attempts: {action.budget.attempts}/{action.budget.max_attempts} · Cost: {action.budget.cost_units_used}/{action.budget.max_cost_units} · Deadline: {action.budget.deadline_at}</p>
    <p>Decision: <EntityReference kind="decision" id={action.provenance.decision_id} available={loadedDecisions} /> · Plan: <EntityReference kind="plan" id={action.provenance.plan_id} available={loadedPlans} />{action.provenance.step_id ? ` / step ${action.provenance.step_id}` : ""} · Journal: <EntityReference kind="journal" id={action.provenance.triggering_event_id} available={loadedJournal} /> · Receipt: <EntityReference kind="receipt" id={action.receipt?.receipt_id ?? null} available={loadedReceipts} /></p>
    {commands.includes("reject") ? <input aria-label={`Reason for ${action.intent_id}`} value={reason} maxLength={500} onChange={(event) => onReason(event.target.value)} placeholder="Optional reason" /> : null}
     {confirmationNeeded && commands.length ? <div><p>Confirm target <span className="mono">{action.intent_id}</span> / digest <span className="mono">{action.preview.digest}</span>. Required phrase: <strong>{action.confirmation?.phrase}</strong></p><input aria-label={`Confirmation for ${action.intent_id}`} value={phrase} onChange={(event) => onPhrase(event.target.value)} /></div> : null}
     <div className="action-row">{commands.map((command) => <Button key={command} disabled={pending || (confirmationNeeded && phrase !== action.confirmation?.phrase)} onClick={() => onRun(action, command)}>{commandLabel(command)}</Button>)}</div>
  </article>;
}

function ActionToolCatalog({ tools }: { tools: ActionTool[] }) {
  return tools.length ? <div className="stack">{tools.map((tool) => <div className="record" key={tool.name}><strong>{tool.name}</strong><p>{tool.effect_code} · Risk: {tool.risk_class} · {tool.executable ? "executable" : "disabled"} · schema {tool.validation_schema_revision.slice(0, 12)}...</p></div>)}</div> : <p className="muted">No action tools.</p>;
}

function RegistryToolCatalog({ tools }: { tools: RegistryTool[] }) {
  return tools.length ? <div className="stack">{tools.map((tool) => <div className="record" key={tool.name}><strong>{tool.name}</strong><p>{tool.description ?? "Description unavailable."} · {tool.tool_type} · {tool.status} · registry only</p></div>)}</div> : <p className="muted">No registry tools.</p>;
}

function formatArgumentSummary(action: OperatorAction): string {
  const summary = action.argument_summary;
  if (summary.kind === "metadata_read") return `Metadata ${summary.namespace} / ${summary.key}`;
  if (summary.kind === "document_search") return `Document scope ${summary.scope_kind} · ${summary.max_results} results · query length ${summary.query_length}`;
  if (summary.kind === "calendar_read") return `Calendar ${summary.starts_at} → ${summary.ends_at} · ${summary.max_results} results`;
  return `${summary.channel} notification · ${summary.title} · ${summary.body_preview}`;
}

function commandLabel(command: OperatorCommand): string {
  return command === "retry_now" ? "Retry now" : command[0].toUpperCase() + command.slice(1);
}

function hasActiveOperatorItems(data: { actions: OperatorAction[] } | undefined): boolean {
  return Boolean(data?.actions.some((action) => ["awaiting_approval", "approved", "retry_pending", "executing"].includes(action.status)));
}

function TrainingNodeRecord({ node, jobs, loadedJobs }: { node: CockpitTrainingNode; jobs: CockpitTrainingJob[]; loadedJobs: Set<string> }) {
  const nodeJobs = jobs.filter((job) => job.worker_node_id === node.node_id).map((job) => job.job_id);
  return <article className="record" id={anchor("training-node", node.node_id)}><div className="metadata-row"><Badge data-tone={node.status === "online" ? "success" : "danger"}>{node.status}</Badge><Badge>{node.role}</Badge><strong>{node.node_id}</strong></div><p>Backend: <span className="mono">{node.backend}</span> · Last contact: {node.last_contact_at ?? "unavailable"}</p><p>GPU: <span className="mono">{node.gpu_name ?? "unavailable"}</span> · CUDA {node.cuda_version ?? "unavailable"} · Driver {node.driver_version ?? "unavailable"}</p><p>Expected: <span className="mono">{node.expected_model_id ?? "unavailable"} / {node.expected_model_revision ?? "unavailable"}</span> · Observed: <span className="mono">{node.observed_model_id ?? "unavailable"} / {node.observed_model_revision ?? "unavailable"}</span> · Match: {node.model_matches_expected === null ? "unavailable" : node.model_matches_expected ? "yes" : "no"}</p><p>Jobs: {nodeJobs.length ? <ReferenceList kind="training-job" ids={nodeJobs} available={loadedJobs} /> : "unavailable"}</p></article>;
}

function TrainingJobRecord({ job, loadedNodes, loadedAdapters }: { job: CockpitTrainingJob; loadedNodes: Set<string>; loadedAdapters: Set<string> }) {
  return <article className="record" id={anchor("training-job", job.job_id)}><div className="metadata-row"><Badge data-tone={job.status === "failed" ? "danger" : job.status === "running" || job.status === "importing" ? "accent" : "neutral"}>{job.status}</Badge><strong>{job.job_id}</strong><span className="mono muted">attempt {job.attempt_id}</span></div><p>Backend: {job.backend ?? "unavailable"} · Worker: <EntityReference kind="training-node" id={job.worker_node_id} available={loadedNodes} /> · Remote: <span className="mono">{job.remote_job_id ?? "unavailable"}</span></p><p>Created: {job.created_at ?? "unavailable"} · Started: {job.started_at ?? "unavailable"} · Completed: {job.completed_at ?? "unavailable"}</p><p>Source events: {job.source_event_start ?? "unavailable"} → {job.source_event_end ?? "unavailable"} · Episodes: {job.selected_episode_count ?? "unavailable"} · Retry: {job.retry_count ?? "unavailable"}</p><p>Transferred: {formatBytes(job.transferred_bytes)} · Failure: <span className="mono">{job.failure_code ?? "none"}</span> · Import: {job.import_status}</p><p>Candidate adapter: <EntityReference kind="adapter-lineage" id={job.candidate_adapter_id} available={loadedAdapters} /> · Bundle: <span className="mono">{shortDigest(job.bundle_digest)}</span> · Result: <span className="mono">{shortDigest(job.result_digest)}</span></p></article>;
}

function AdapterLineageRecord({ adapter, loadedJobs, loadedNodes, loadedEvaluations, loadedJournal }: { adapter: CockpitAdapterLineage; loadedJobs: Set<string>; loadedNodes: Set<string>; loadedEvaluations: Set<string>; loadedJournal: Set<string> }) {
  return <article className="record" id={anchor("adapter-lineage", adapter.adapter_id)}><div className="metadata-row"><Badge data-tone={adapter.active ? "success" : adapter.rollback_candidate ? "warning" : "neutral"}>{adapter.status}</Badge><strong>{adapter.adapter_id}</strong><span className="mono muted">{shortDigest(adapter.adapter_hash)}</span></div><p>Base revision: <span className="mono">{adapter.base_model_id ?? "unavailable"} / {adapter.base_model_revision ?? "unavailable"}</span> · Parent: <span className="mono muted">{adapter.parent_adapter_id ?? "none"}</span></p><p>Training job: <EntityReference kind="training-job" id={adapter.training_job_id} available={loadedJobs} /> · Training node: <EntityReference kind="training-node" id={adapter.training_node_id} available={loadedNodes} /></p><p>Submitter: <span className="mono">{adapter.submitted_by_node_id ?? "unavailable"}</span> · Importer: <span className="mono">{adapter.imported_by_node_id ?? "unavailable"}</span></p><p>Evaluation: <EvaluationReference id={adapter.evaluation_id} available={loadedEvaluations} /> · {adapter.evaluation_status} · Approval: {adapter.approved ? "approved" : "not approved"} · Active: {adapter.active ? "yes" : "no"} · Rollback candidate: {adapter.rollback_candidate ? "yes" : "no"}</p><p>Activation event: <EntityReference kind="journal" id={adapter.activation_event_id} available={loadedJournal} />{adapter.activation_event_sequence ? ` @${adapter.activation_event_sequence}` : ""} · Rollback event: <EntityReference kind="journal" id={adapter.rollback_event_id} available={loadedJournal} />{adapter.rollback_event_sequence ? ` @${adapter.rollback_event_sequence}` : ""}</p></article>;
}

function OutboxSummary({ pendingCount, criticalCount, visibleMessages, loadedGoals, loadedPlans, loadedDecisions, loadedCommitments, loadedActions }: { pendingCount: number; criticalCount: number; visibleMessages: CockpitOutboxMessage[]; loadedGoals: Set<string>; loadedPlans: Set<string>; loadedDecisions: Set<string>; loadedCommitments: Set<string>; loadedActions: Set<string> }) {
  return <><div className="metric-grid"><Metric label="Pending" value={String(pendingCount)} /><Metric label="Critical" value={String(criticalCount)} /></div><div className="stack">{visibleMessages.map((message) => <article className="record" id={anchor("outbox", message.message_id)} key={message.message_id}><div className="metadata-row"><Badge data-tone={message.urgency === "critical" ? "danger" : "neutral"}>{message.urgency}</Badge><Badge>{message.delivery_status}</Badge><strong>{message.title}</strong></div><p>Goal: <EntityReference kind="goal" id={message.references.goal_id} available={loadedGoals} /> · Plan: <EntityReference kind="plan" id={message.references.plan_id} available={loadedPlans} /></p><p>Decision: <EntityReference kind="decision" id={message.references.decision_id} available={loadedDecisions} /> · Commitment: <EntityReference kind="commitment" id={message.references.commitment_id} available={loadedCommitments} /> · Action: <EntityReference kind="action" id={message.references.action_id} available={loadedActions} /></p></article>)}</div></>;
}

function JournalEntry({ record, actions, loadedJournal, loadedActions, loadedReceipts }: { record: JournalRecord; actions: CockpitActionTrace[]; loadedJournal: Set<string>; loadedActions: Set<string>; loadedReceipts: Set<string> }) {
  const trace = actions.find((item) => item.receipt?.event_id === record.event_id);
  return <article className="record" id={anchor("journal", record.event_id)}><div className="metadata-row"><Badge>{record.lifecycle}</Badge><strong>{record.event_type}</strong></div><p className="mono muted">{record.event_id} · {new Date(record.timestamp).toLocaleString()}</p><p>Target: <span className="mono">{record.target ?? "unavailable"}</span> · Category: {record.failure_category ?? "none"}</p><p>Causation: <EntityReference kind="journal" id={record.causation_id} available={loadedJournal} /> · Correlation: <span className="mono">{record.correlation_id ?? "unavailable"}</span></p>{trace?.receipt ? <p>Action: <EntityReference kind="action" id={trace.intent_id} available={loadedActions} /> · Receipt: <EntityReference kind="receipt" id={trace.receipt.receipt_id} available={loadedReceipts} /></p> : null}</article>;
}

function EntityReference({ kind, id, available }: { kind: string; id: string | null; available: Set<string> }) {
  if (id === null) return <>unavailable</>;
  return available.has(id) ? <Link className="entity-link mono" href={`#${anchor(kind, id)}`}>{id}</Link> : <span className="mono muted">{id}</span>;
}

function EvaluationReference({ id, available }: { id: string | null; available: Set<string> }) {
  if (id === null) return <>unavailable</>;
  return available.has(id) ? <Link className="entity-link mono" href={evaluationHref(id)}>{id}</Link> : <span className="mono muted">{id}</span>;
}

function ReferenceList({ kind, ids, available }: { kind: string; ids: string[]; available: Set<string> }) {
  return <>{ids.map((id, index) => <span key={id}>{index ? ", " : ""}<EntityReference kind={kind} id={id} available={available} /></span>)}</>;
}

function anchor(kind: string, id: string): string {
  return `${kind}-${encodeURIComponent(id)}`;
}

function isCurrentGoal(goal: Goal): boolean {
  return goal.status === "active" || goal.status === "candidate";
}

function isCurrentCommitment(commitment: Commitment): boolean {
  return commitment.status === "active" || commitment.status === "proposed" || commitment.status === "renegotiating";
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function shortDigest(value: string | null): string {
  return value === null ? "unavailable" : `${value.slice(0, 12)}...`;
}

function formatBytes(value: number | null): string {
  if (value === null) return "unavailable";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KiB`;
  return `${Math.round(value / (1024 * 1024))} MiB`;
}

function actionTone(status: CockpitActionTrace["status"]): string {
  if (status === "awaiting_approval" || status === "retry_pending") return "warning";
  if (status === "failed" || status === "rejected" || status === "cancelled") return "danger";
  if (status === "succeeded") return "success";
  return "neutral";
}
