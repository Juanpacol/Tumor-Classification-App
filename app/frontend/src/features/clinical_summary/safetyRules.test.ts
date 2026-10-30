import { SAMPLE_STUDIES } from "@/services/api/sampleData";
import { safetyChecks } from "./safetyRules";

function checks(id: string) {
  const study = SAMPLE_STUDIES.find((candidate) => candidate.id === id);
  if (!study) throw new Error(`no sample study ${id}`);
  return safetyChecks(study);
}

function states(id: string) {
  return Object.fromEntries(
    checks(id).map((check) => [check.key, check.state]),
  );
}

describe("safetyChecks", () => {
  it("passes everything for a confident study with an agreeing report", () => {
    expect(states("study-confident")).toEqual({
      quality: "pass",
      confidence: "pass",
      review: "note",
      agreement: "pass",
      wording: "pass",
    });
  });

  it("flags the low confidence and requires expert review", () => {
    expect(states("study-uncertain")).toMatchObject({
      quality: "pass",
      confidence: "attention",
      review: "attention",
    });
    const review = checks("study-uncertain").find((c) => c.key === "review");
    expect(review?.label).toBe("Expert review required");
  });

  it("flags a report the assistant could not confirm", () => {
    expect(states("study-uncertain").agreement).toBe("attention");
  });

  it("waits for the report of a completed study that has none", () => {
    expect(states("study-no-report")).toMatchObject({ agreement: "pending" });
    expect(states("study-no-report").wording).toBeUndefined();
  });

  it("explains a rejected study with its reason and nothing about a result", () => {
    const list = checks("study-rejected");
    expect(list.map((check) => check.key)).toEqual(["quality", "review"]);
    expect(list[0]).toMatchObject({
      state: "attention",
      detail: "The image is too noisy to classify.",
    });
  });

  it("does not invent checks for a study that is still running", () => {
    const list = checks("study-processing");
    expect(list.map((check) => check.key)).toEqual(["quality", "review"]);
    expect(list[0]?.state).toBe("pending");
  });

  it("states the image quality figures it really has", () => {
    const quality = checks("study-confident")[0];
    expect(quality?.detail).toBe("Signal-to-noise 22.4 · entropy 6.10");
  });

  it("never marks a standing rule as done", () => {
    expect(states("study-confident").review).toBe("note");
  });
});
