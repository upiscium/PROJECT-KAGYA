export function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

export function statusTone(status: string): "neutral" | "success" | "danger" | "accent" {
  if (status === "active" || status === "approved") return "success";
  if (status === "rejected" || status === "archived") return "danger";
  if (status === "candidate" || status === "trial_active") return "accent";
  return "neutral";
}
