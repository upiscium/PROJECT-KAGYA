import { formatNumber } from "@/lib/format";
import type { Emotion } from "@/lib/api";

export function EmotionMeter({ emotion }: { emotion: Emotion }) {
  return (
    <div className="emotion-meter" aria-label="Emotion state">
      <span>Valence {formatNumber(emotion.valence)}</span>
      <span>Arousal {formatNumber(emotion.arousal)}</span>
    </div>
  );
}
