import { Alert } from "@/features/portal/components";
import { formatConfidence } from "@/features/portal/labels";

/** Shown when the model flags a prediction for expert triage. */
export function UncertaintyAlert({ confidence }: { confidence: number }) {
  return (
    <Alert tone="warning" title="Needs expert review">
      The model is not confident enough in this prediction (
      {formatConfidence(confidence)}). Do not rely on it without an expert
      reading of the image.
    </Alert>
  );
}
