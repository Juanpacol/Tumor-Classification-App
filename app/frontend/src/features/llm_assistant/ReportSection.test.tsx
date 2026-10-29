import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError, type ApiClient, type Report } from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { SAMPLE_REPORT, SAMPLE_STUDIES } from "@/services/api/sampleData";
import { REPORT_DISCLAIMER } from "./ReportView";
import { ReportSection } from "./ReportSection";

function study(id: string) {
  const found = SAMPLE_STUDIES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no sample study ${id}`);
  return found;
}

function reportWith(agreement: "agrees" | "disagrees" | "uncertain"): Report {
  return {
    ...SAMPLE_REPORT,
    content: { ...SAMPLE_REPORT.content!, ai_agreement: agreement },
  };
}

function clientWith(generateReport: ApiClient["generateReport"]): ApiClient {
  return { ...createMockClient(), generateReport };
}

describe("ReportSection", () => {
  it("shows the five sections and the disclaimer without asking for a new report", () => {
    const generateReport = jest.fn();
    render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-confident")}
      />,
    );
    for (const name of [
      "Findings",
      "Visual observations",
      "Impression",
      "Differential considerations",
      "Suggested next steps",
    ]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
    expect(
      screen.getByText("Review the images with a radiologist."),
    ).toBeInTheDocument();
    expect(screen.getByText(REPORT_DISCLAIMER)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(generateReport).not.toHaveBeenCalled();
  });

  it.each(["disagrees", "uncertain"] as const)(
    "warns about a discrepancy when the assistant %s",
    (agreement) => {
      render(
        <ReportSection
          client={clientWith(jest.fn())}
          study={{ ...study("study-confident"), report: reportWith(agreement) }}
        />,
      );
      expect(screen.getByRole("alert")).toHaveTextContent("do not agree");
      expect(screen.getByRole("alert")).toHaveTextContent("specialist");
    },
  );

  it("requests the report once, with a loading state", async () => {
    const generateReport = jest.fn(async () => SAMPLE_REPORT);
    const { rerender } = render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-no-report")}
      />,
    );
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Generating the report",
    );
    expect(
      await screen.findByRole("heading", { name: "Findings" }),
    ).toBeInTheDocument();
    rerender(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-no-report")}
      />,
    );
    expect(generateReport).toHaveBeenCalledTimes(1);
  });

  it("requests the report once even when effects run twice in development", async () => {
    const generateReport = jest.fn(async () => SAMPLE_REPORT);
    render(
      <StrictMode>
        <ReportSection
          client={clientWith(generateReport)}
          study={study("study-no-report")}
        />
      </StrictMode>,
    );
    await screen.findByRole("heading", { name: "Findings" });
    expect(generateReport).toHaveBeenCalledTimes(1);
  });

  it("shows the error with a Retry and does not retry on its own", async () => {
    const generateReport = jest
      .fn<ReturnType<ApiClient["generateReport"]>, []>()
      .mockRejectedValueOnce(
        new ApiError(
          "LLM_UNAVAILABLE",
          "The report service is not available.",
          503,
        ),
      )
      .mockResolvedValueOnce(SAMPLE_REPORT);
    render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-report-unavailable")}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The report service is not available.",
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(generateReport).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByRole("heading", { name: "Findings" }),
    ).toBeInTheDocument();
    expect(generateReport).toHaveBeenCalledTimes(2);
  });

  it("treats a report that did not complete as an error", async () => {
    const failed: Report = {
      ...SAMPLE_REPORT,
      status: "failed",
      content: null,
    };
    render(
      <ReportSection
        client={clientWith(async () => failed)}
        study={study("study-no-report")}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The report could not be generated.",
    );
  });

  it.each(["study-rejected", "study-failed"])("is hidden for %s", (id) => {
    render(<ReportSection client={clientWith(jest.fn())} study={study(id)} />);
    expect(
      screen.queryByRole("heading", { name: "AI clinical report" }),
    ).not.toBeInTheDocument();
  });

  it("waits for the analysis without asking for a report", () => {
    const generateReport = jest.fn();
    render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-processing")}
      />,
    );
    expect(
      screen.getByText("Available when the analysis finishes."),
    ).toBeInTheDocument();
    expect(generateReport).not.toHaveBeenCalled();
  });

  it("asks for confirmation before regenerating", async () => {
    const generateReport = jest.fn(async () => reportWith("uncertain"));
    render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-confident")}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Regenerate report" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(generateReport).not.toHaveBeenCalled();

    await userEvent.click(
      screen.getByRole("button", { name: "Regenerate report" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
    await waitFor(() => expect(generateReport).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("alert")).toHaveTextContent("do not agree");
  });

  it("keeps the current report when regenerating fails", async () => {
    const generateReport = jest.fn(async () => {
      throw new ApiError(
        "LLM_UNAVAILABLE",
        "The report service is not available.",
        503,
      );
    });
    render(
      <ReportSection
        client={clientWith(generateReport)}
        study={study("study-confident")}
      />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Regenerate report" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not available");
    expect(
      screen.getByRole("heading", { name: "Findings" }),
    ).toBeInTheDocument();
  });
});
