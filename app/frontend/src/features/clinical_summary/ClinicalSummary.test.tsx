import { render, screen, within } from "@testing-library/react";
import { SAMPLE_STUDIES } from "@/services/api/sampleData";
import { ClinicalSummary } from "./ClinicalSummary";

function study(id: string) {
  const found = SAMPLE_STUDIES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no sample study ${id}`);
  return found;
}

describe("ClinicalSummary", () => {
  it("raises no alert when the model is confident", () => {
    render(<ClinicalSummary study={study("study-confident")} />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText("Needs expert review")).not.toBeInTheDocument();
  });

  it("lists the safety checks of the study", () => {
    render(<ClinicalSummary study={study("study-confident")} />);
    expect(
      screen.getByRole("region", { name: "Safety checks" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Image quality check passed")).toBeInTheDocument();
  });

  it("lists the four probabilities, highest first", () => {
    render(<ClinicalSummary study={study("study-confident")} />);
    const meters = screen.getAllByRole("meter");
    expect(meters).toHaveLength(4);
    expect(meters[0]).toHaveAccessibleName("Meningioma probability");
    expect(meters[0]).toHaveAttribute("aria-valuenow", "93");
    const values = meters.map((meter) =>
      Number(meter.getAttribute("aria-valuenow")),
    );
    expect(values).toEqual([...values].sort((a, b) => b - a));
  });

  it("shows the quality metrics, skull stripping and model version", () => {
    render(<ClinicalSummary study={study("study-confident")} />);
    expect(screen.getByText("22.4")).toBeInTheDocument();
    expect(screen.getByText("6.10")).toBeInTheDocument();
    expect(screen.getByText("Applied")).toBeInTheDocument();
    expect(screen.getByText("mock-0.1.0")).toBeInTheDocument();
  });

  it("warns and withholds the class when the model needs expert review", () => {
    render(<ClinicalSummary study={study("study-uncertain")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Needs expert review");
    expect(screen.getByRole("alert")).toHaveTextContent("52%");
    expect(
      screen.getAllByText("Needs expert review", { selector: "strong" }),
    ).toHaveLength(1);
    expect(screen.getByText("Low confidence flagged")).toBeInTheDocument();
    expect(screen.getAllByRole("meter")).toHaveLength(4);
    expect(
      within(screen.getAllByRole("meter")[0]!.parentElement!).getByText(
        "Glioma",
      ),
    ).toBeInTheDocument();
  });

  it("shows the rejection reason instead of a summary", () => {
    render(<ClinicalSummary study={study("study-rejected")} />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The image is too noisy to classify.",
    );
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });

  it("falls back to a generic reason when none is given", () => {
    render(
      <ClinicalSummary
        study={{ ...study("study-rejected"), rejection_reason: null }}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "did not pass the quality check",
    );
  });

  it("reports a failed analysis", () => {
    render(<ClinicalSummary study={study("study-failed")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("The analysis failed");
  });

  it("says the analysis is running while there is no prediction yet", () => {
    render(<ClinicalSummary study={study("study-processing")} />);
    expect(screen.getByRole("status")).toHaveTextContent("still running");
  });

  it("says so when a completed study has no prediction", () => {
    render(
      <ClinicalSummary
        study={{ ...study("study-confident"), prediction: null }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "No prediction is available",
    );
  });
});
