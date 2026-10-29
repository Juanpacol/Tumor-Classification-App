import type { Prediction, Report, StudyDetail, TumorLabel } from "./types";

function prediction(
  label: TumorLabel,
  confidence: number,
  needsExpertTriage: boolean,
): Prediction {
  const rest = (1 - confidence) / 3;
  const probabilities = {
    glioma: rest,
    meningioma: rest,
    pituitary: rest,
    no_tumor: rest,
    [label]: confidence,
  } as Record<TumorLabel, number>;
  return {
    label,
    probabilities,
    confidence,
    needs_expert_triage: needsExpertTriage,
    quality: { snr: 22.4, entropy: 6.1, skull_stripped: true },
    top_features: [
      { name: "glcm_contrast", value: 0.42, z_score: 1.8 },
      { name: "lbp_uniformity", value: 0.31, z_score: -1.2 },
    ],
    model_version: "mock-0.1.0",
  };
}

const COMPLETED_REPORT: Report = {
  status: "completed",
  model: "mock",
  created_at: "2026-10-28T10:05:00Z",
  error: null,
  content: {
    findings: "Sample findings for the mock study.",
    visual_observations: "Sample visual observations.",
    impression: "Sample impression, to be confirmed by a radiologist.",
    differential_considerations: "Sample differential considerations.",
    suggested_next_steps: ["Review the images with a radiologist."],
    ai_agreement: "agrees",
  },
};

const base = { clinical_notes: null, rejection_reason: null };

/** One study per state the portal has to render. */
export const SAMPLE_STUDIES: StudyDetail[] = [
  {
    ...base,
    id: "study-confident",
    status: "completed",
    created_at: "2026-10-28T10:00:00Z",
    prediction: prediction("meningioma", 0.93, false),
    report: COMPLETED_REPORT,
  },
  {
    ...base,
    id: "study-uncertain",
    status: "completed",
    created_at: "2026-10-27T10:00:00Z",
    prediction: prediction("glioma", 0.52, true),
    report: {
      ...COMPLETED_REPORT,
      content: {
        ...COMPLETED_REPORT.content!,
        ai_agreement: "uncertain",
      },
    },
  },
  {
    ...base,
    id: "study-no-report",
    status: "completed",
    created_at: "2026-10-26T10:00:00Z",
    prediction: prediction("no_tumor", 0.88, false),
    report: null,
  },
  {
    ...base,
    id: "study-processing",
    status: "processing",
    created_at: "2026-10-25T10:00:00Z",
    prediction: null,
    report: null,
  },
  {
    ...base,
    id: "study-rejected",
    status: "rejected",
    created_at: "2026-10-24T10:00:00Z",
    rejection_reason: "The image is too noisy to classify.",
    prediction: null,
    report: null,
  },
  {
    ...base,
    id: "study-failed",
    status: "failed",
    created_at: "2026-10-23T10:00:00Z",
    prediction: null,
    report: null,
  },
];
