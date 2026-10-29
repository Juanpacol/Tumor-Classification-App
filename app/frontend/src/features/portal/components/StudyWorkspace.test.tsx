import { render, screen } from "@testing-library/react";
import { createMockClient } from "@/services/api/mockClient";
import { StudyWorkspace } from "./StudyWorkspace";

jest.mock("@/features/dicom_viewer/StudyViewer", () => ({
  StudyViewer: ({ imageUrl }: { imageUrl: string }) => (
    <div data-testid="viewer">{imageUrl}</div>
  ),
}));

describe("StudyWorkspace", () => {
  it("lays out the viewer and the areas for summary, report and chat", async () => {
    render(<StudyWorkspace id="study-confident" client={createMockClient()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading study");
    expect(await screen.findByTestId("viewer")).toHaveTextContent(
      "/mock/study-confident.png",
    );
    for (const name of ["Clinical summary", "AI clinical report", "Chat"]) {
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    }
    expect(screen.getByRole("link", { name: /All studies/ })).toHaveAttribute(
      "href",
      "/studies",
    );
  });

  it("explains a study that does not exist", async () => {
    render(<StudyWorkspace id="nope" client={createMockClient()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The study does not exist.",
    );
    expect(screen.queryByTestId("viewer")).not.toBeInTheDocument();
  });
});
