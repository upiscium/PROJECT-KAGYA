import { Card, CardTitle } from "@/components/ui/card";

export default function EvaluationsPage() {
  return (
    <div className="page">
      <header><h1 className="page-title">Evaluations</h1><p className="page-subtitle">Evaluation result JSON is produced by adapter evaluation endpoints.</p></header>
      <Card><CardTitle>Current Scope</CardTitle><p>Use the Adapters page to trigger evaluations. Detailed evaluation browsing will attach to persisted result paths once backend listing endpoints are added.</p></Card>
    </div>
  );
}
