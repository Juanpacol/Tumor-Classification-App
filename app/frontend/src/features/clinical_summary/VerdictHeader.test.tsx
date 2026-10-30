import { render, screen } from "@testing-library/react";
import { SAMPLE_STUDIES } from "@/services/api/sampleData";
import { VerdictHeader } from "./VerdictHeader";

function study(id: string) {
  const found = SAMPLE_STUDIES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no sample study ${id}`);
  return found;
}

describe("VerdictHeader", () => {
  it("names the class and shows the confidence ring when the model is confident", () => {
    render(<VerdictHeader study={study("study-confident")} />);
    expect(screen.getByText("Meningioma")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "93% confidence" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Model mock-0.1.0")).toBeInTheDocument();
  });

  it("does not name a class when the model asks for expert review", () => {
    render(<VerdictHeader study={study("study-uncertain")} />);
    expect(screen.getByText("Needs expert review")).toBeInTheDocument();
    expect(screen.queryByText("Glioma")).not.toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "52% confidence" }),
    ).toBeInTheDocument();
  });

  it.each([
    ["study-rejected", "Rejected"],
    ["study-failed", "Failed"],
    ["study-processing", "In progress"],
  ])("says %s is %s, without a ring", (id, label) => {
    render(<VerdictHeader study={study(id)} />);
    expect(screen.getByText(label)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
