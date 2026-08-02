export const queryKeys = {
  cockpit: {
    runtime: ["cockpit", "runtime"] as const,
    emotion: ["cockpit", "emotion"] as const,
    workingMemory: ["cockpit", "working-memory"] as const,
    contexts: ["cockpit", "contexts"] as const,
    goals: ["cockpit", "goals"] as const,
    commitments: ["cockpit", "commitments"] as const,
    plans: ["cockpit", "plans"] as const,
    decisions: ["cockpit", "decisions"] as const,
    outbox: ["cockpit", "outbox"] as const,
    actions: ["cockpit", "actions"] as const,
    training: ["cockpit", "training"] as const,
    evaluations: ["cockpit", "evaluations"] as const,
    journal: ["cockpit", "journal"] as const,
    adapters: ["cockpit", "adapters"] as const,
    actionOperator: ["cockpit", "action-operator"] as const,
    restoreSummary: ["cockpit", "restore-summary"] as const,
    restorePreview: (targetSequence: number) => ["cockpit", "restore-preview", targetSequence] as const,
  },
  decisionExplanations: ["decision-explanations"] as const,
  outbox: ["outbox"] as const,
};

export const actionMutationInvalidationKeys = [
  queryKeys.cockpit.actionOperator,
  queryKeys.cockpit.actions,
  queryKeys.cockpit.decisions,
  queryKeys.cockpit.plans,
  queryKeys.cockpit.outbox,
  queryKeys.cockpit.journal,
  queryKeys.decisionExplanations,
  queryKeys.outbox,
] as const;

export const operatorRestoreInvalidationKeys = [
  ...Object.values(queryKeys.cockpit).filter((key) => Array.isArray(key)),
  queryKeys.decisionExplanations,
  queryKeys.outbox,
] as const;
