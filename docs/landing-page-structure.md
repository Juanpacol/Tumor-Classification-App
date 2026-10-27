# Landing Page Structure (Hackathon Pitch)

Content strategy for the public-facing landing page. Goal: sell the product to hackathon judges and voters, not explain the ML architecture in depth. The technical architecture lives in `CLAUDE.md` and the architecture report; this doc is about what a visitor sees and reads.

## 1. Hero

- One benefit-driven headline, not a technology statement.
  - Example direction: "Give every hospital access to an expert radiologist, in minutes."
- Supporting visual: a Grad-CAM heatmap over an MRI scan. This is the most visually distinctive asset the product has.
- No deep technical terms here (no "CNN", "radiomics", "fusion").

## 2. Problem → Solution

- Lead with one hard number about radiologist shortage in underserved regions.
- Follow immediately with the one-line solution: "Ether gives a trustworthy second opinion in minutes."
- Business framing, not architecture framing.

## 3. How It Works (process transparency)

A simple, visual timeline the visitor can follow, framed as "how the model reasoned," not as a system diagram:

1. Extracts texture and structural features from the scan.
2. Analyzes visual patterns with a deep learning model.
3. Combines both signals into a single finding.
4. Shows exactly which part of the brain drove the conclusion (Grad-CAM overlay).

Purpose: make the model's reasoning feel auditable, not a black box.

## 4. Safety Rubric / Trust Section

A visible checklist-style card, the product's "seal of guarantee." Draws directly from the existing guardrails in the architecture (see `CLAUDE.md` → Failure Modes, Edge Cases, and Guardrails):

- Scan quality check before any prediction is attempted (entropy/SNR check).
- Confidence threshold enforced; low-confidence cases are flagged, not force-classified.
- Cases below threshold are marked `NEEDS_EXPERT_TRIAGE` and routed to mandatory human review.
- The system never outputs a definitive diagnosis — only decision-support language.

Framing: "We know when not to trust ourselves, and we tell you." This is the differentiator most health-hackathon projects don't show.

## 5. Demo / Proof

- Live demo or recorded GIF of the DICOM viewer with the heatmap overlay in action.
- This sells harder than any copy on the page.

## 6. Differentiators (secondary section, for technical judges who dig deeper)

- Explainability (Grad-CAM) as a trust mechanism, not a feature checkbox.
- Data sovereignty: works with an on-premise, open-source LLM (Llama via Ollama/vLLM) instead of a locked-in cloud API — relevant for clinics without reliable cloud access.
- Hybrid ML architecture (classical radiomics + deep learning) for better generalization on small, heterogeneous datasets — one sentence, not a diagram, unless a judge asks.

## 7. Call to Action

- SaaS-style CTA: "Request a demo" / "See it in action."
- Team/stack credits go last, brief.

## Tone Rules

- Speak to the hospital/doctor's outcome, not the engineering implementation.
- Never claim diagnostic certainty; always "decision support," "second opinion," "flagged for review."
- Numbers and specifics over adjectives.
