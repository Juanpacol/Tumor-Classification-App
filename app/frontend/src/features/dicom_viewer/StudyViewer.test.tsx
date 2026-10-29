import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StudyViewer } from "./StudyViewer";
import {
  ViewerUnavailableError,
  type CreateViewer,
  type Viewer,
} from "./viewerEngine";

const mockCreateViewer = jest.fn<ReturnType<CreateViewer>, []>();

jest.mock("./viewerEngine", () => ({
  ...jest.requireActual("./viewerEngine"),
  createViewer: () => mockCreateViewer(),
}));

describe("StudyViewer", () => {
  it("offers the tools and changes the active one", async () => {
    const viewer: Viewer = {
      setTool: jest.fn(),
      reset: jest.fn(),
      destroy: jest.fn(),
    };
    mockCreateViewer.mockResolvedValue(viewer);
    render(<StudyViewer imageUrl="http://api.test/image" />);
    const zoom = await screen.findByRole("button", { name: "Zoom" });
    expect(
      screen.getByRole("button", { name: "Window/level" }),
    ).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(zoom);
    expect(viewer.setTool).toHaveBeenCalledWith("zoom");
    expect(zoom).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(viewer.reset).toHaveBeenCalled();
  });

  it("shows a message when the image is not available", async () => {
    mockCreateViewer.mockRejectedValue(
      new ViewerUnavailableError("image_not_found"),
    );
    render(<StudyViewer imageUrl="http://api.test/image" />);
    expect(
      await screen.findByText("The image for this study is not available."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Zoom" }),
    ).not.toBeInTheDocument();
  });

  it("explains a browser without WebGL", async () => {
    mockCreateViewer.mockRejectedValue(new ViewerUnavailableError("no_webgl"));
    render(<StudyViewer imageUrl="http://api.test/image" />);
    expect(
      await screen.findByText(/WebGL is not available/),
    ).toBeInTheDocument();
  });
});
