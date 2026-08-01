const EVALUATION_PREFIX = "evaluation-";

export function evaluationAnchor(evaluationId: string): string {
  return `${EVALUATION_PREFIX}${encodeURIComponent(evaluationId)}`;
}

export function evaluationHref(evaluationId: string): string {
  return `/evaluations#${evaluationAnchor(evaluationId)}`;
}

export function evaluationIdFromHash(hash: string): string | null {
  const prefix = `#${EVALUATION_PREFIX}`;
  if (!hash.startsWith(prefix)) return null;
  try {
    return decodeURIComponent(hash.slice(prefix.length));
  } catch {
    return null;
  }
}
