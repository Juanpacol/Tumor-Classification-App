import type { Prediction } from "@/services/api";
import styles from "./QualityMetrics.module.css";

/** Image quality, skull stripping and model version behind the prediction. */
export function QualityMetrics({ prediction }: { prediction: Prediction }) {
  const { quality, model_version } = prediction;
  return (
    <dl className={styles.list}>
      <dt>Signal-to-noise ratio</dt>
      <dd>{quality.snr.toFixed(1)}</dd>
      <dt>Entropy</dt>
      <dd>{quality.entropy.toFixed(2)}</dd>
      <dt>Skull stripping</dt>
      <dd>{quality.skull_stripped ? "Applied" : "Not applied"}</dd>
      <dt>Model version</dt>
      <dd>{model_version}</dd>
    </dl>
  );
}
