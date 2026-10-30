import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ApiClient } from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { SAMPLE_REPORT, SAMPLE_STUDIES } from "@/services/api/sampleData";
import { AnalysisPanel } from "./AnalysisPanel";

function study(id: string) {
  const found = SAMPLE_STUDIES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no sample study ${id}`);
  return found;
}

function tabNames() {
  return screen.getAllByRole("tab").map((tab) => tab.textContent);
}

describe("AnalysisPanel", () => {
  it("offers the summary, the report and the chat for a completed study", () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-confident")}
      />,
    );
    expect(tabNames()).toEqual(["Summary", "Report", "Chat"]);
    expect(screen.getByRole("tab", { name: "Summary" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("has nothing to report on or chat about for a rejected study", () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-rejected")}
      />,
    );
    expect(tabNames()).toEqual(["Summary"]);
  });

  it("keeps the chat for completed studies only", () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-processing")}
      />,
    );
    expect(tabNames()).toEqual(["Summary", "Report"]);
  });

  it("shows the verdict above the tabs", () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-uncertain")}
      />,
    );
    expect(screen.getAllByText("Needs expert review").length).toBeGreaterThan(
      0,
    );
    expect(
      screen.getByRole("img", { name: "52% confidence" }),
    ).toBeInTheDocument();
  });

  it("does not ask the model for a report until the Report tab is opened", async () => {
    const generateReport = jest.fn(async () => SAMPLE_REPORT);
    const client: ApiClient = { ...createMockClient(), generateReport };
    render(<AnalysisPanel client={client} study={study("study-no-report")} />);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(generateReport).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("tab", { name: "Report" }));
    await waitFor(() => expect(generateReport).toHaveBeenCalledTimes(1));
    await userEvent.click(screen.getByRole("tab", { name: "Summary" }));
    await userEvent.click(screen.getByRole("tab", { name: "Report" }));
    expect(generateReport).toHaveBeenCalledTimes(1);
  });

  it("moves between tabs with the arrow keys", async () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-confident")}
      />,
    );
    screen.getByRole("tab", { name: "Summary" }).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Report" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Report" })).toHaveFocus();
    await userEvent.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("keeps only the open tab visible", async () => {
    render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-confident")}
      />,
    );
    expect(screen.getByRole("tabpanel", { name: "Summary" })).toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: "Chat" }));
    expect(screen.getByRole("tabpanel", { name: "Chat" })).toBeVisible();
    expect(
      screen.queryByRole("tabpanel", { name: "Summary" }),
    ).not.toBeInTheDocument();
  });

  it("changes the drawing behind the whole panel with the tab", async () => {
    const { container } = render(
      <AnalysisPanel
        client={createMockClient()}
        study={study("study-confident")}
      />,
    );
    const backdrop = container.querySelector("[data-tab]");
    expect(backdrop).toHaveAttribute("data-tab", "summary");
    await userEvent.click(screen.getByRole("tab", { name: "Chat" }));
    expect(backdrop).toHaveAttribute("data-tab", "chat");
  });
});
