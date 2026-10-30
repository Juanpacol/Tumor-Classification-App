import { render, screen } from "@testing-library/react";
import { createMockClient } from "@/services/api/mockClient";
import { StudyWorkspace } from "./StudyWorkspace";

jest.mock("@/features/dicom_viewer/StudyViewer", () => ({
  StudyViewer: ({
    imageUrl,
    flagged,
  }: {
    imageUrl: string;
    flagged?: boolean;
  }) => (
    <div data-testid="viewer" data-flagged={String(flagged)}>
      {imageUrl}
    </div>
  ),
}));

describe("StudyWorkspace", () => {
  it("lays out the viewer and the areas for summary, report and chat", async () => {
    render(<StudyWorkspace id="study-confident" client={createMockClient()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading study");
    expect(await screen.findByTestId("viewer")).toHaveTextContent(
      "/mock/study-confident.png",
    );
    for (const name of ["Summary", "Report", "Chat"]) {
      expect(screen.getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(screen.getByRole("link", { name: /All studies/ })).toHaveAttribute(
      "href",
      "/studies",
    );
  });

  it("tells the viewer when the study needs expert review", async () => {
    render(<StudyWorkspace id="study-uncertain" client={createMockClient()} />);
    expect(await screen.findByTestId("viewer")).toHaveAttribute(
      "data-flagged",
      "true",
    );
  });

  it("fills the summary area with the prediction", async () => {
    render(<StudyWorkspace id="study-uncertain" client={createMockClient()} />);
    expect(
      await screen.findAllByText("Needs expert review", { selector: "strong" }),
    ).toHaveLength(2);
  });

  it("explains a study that does not exist", async () => {
    render(<StudyWorkspace id="nope" client={createMockClient()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The study does not exist.",
    );
    expect(screen.queryByTestId("viewer")).not.toBeInTheDocument();
  });
});
