import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StudyViewer } from "./StudyViewer";
import { ZOOM_STEP } from "./zoom";
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

function fakeViewer(): Viewer {
  return {
    setTool: jest.fn(),
    zoomBy: jest.fn(),
    onZoom: jest.fn(() => jest.fn()),
    reset: jest.fn(),
    destroy: jest.fn(),
  };
}

describe("StudyViewer", () => {
  it("zooms in and out from the buttons", async () => {
    const viewer = fakeViewer();
    mockCreateViewer.mockResolvedValue(viewer);
    render(<StudyViewer imageUrl="http://api.test/image" />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Zoom in" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(viewer.zoomBy).toHaveBeenNthCalledWith(1, ZOOM_STEP);
    expect(viewer.zoomBy).toHaveBeenNthCalledWith(2, 1 / ZOOM_STEP);
  });

  it("shows the zoom level and follows the viewer", async () => {
    let announce: (zoom: number) => void = () => {};
    const viewer: Viewer = {
      ...fakeViewer(),
      onZoom: jest.fn((listener) => {
        announce = listener;
        return jest.fn();
      }),
    };
    mockCreateViewer.mockResolvedValue(viewer);
    render(<StudyViewer imageUrl="http://api.test/image" />);
    expect(await screen.findByLabelText("Zoom level")).toHaveTextContent(
      "100%",
    );
    act(() => announce(1.56));
    expect(screen.getByLabelText("Zoom level")).toHaveTextContent("156%");
  });

  it("tells the cursor which tool is active and says how to use it", async () => {
    mockCreateViewer.mockResolvedValue(fakeViewer());
    const { container } = render(
      <StudyViewer imageUrl="http://api.test/image" />,
    );
    await screen.findByRole("button", { name: "Pan" });
    const canvas = container.querySelector("[data-tool]");
    expect(canvas).toHaveAttribute("data-tool", "windowLevel");
    expect(screen.getByText(/Drag to adjust the contrast/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Pan" }));
    expect(canvas).toHaveAttribute("data-tool", "pan");
    expect(screen.getByText(/Drag to move the image/)).toBeInTheDocument();
    expect(screen.getByText(/Scroll to zoom/)).toBeInTheDocument();
  });

  it("offers the tools and changes the active one", async () => {
    const viewer = fakeViewer();
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

  it("says so when the viewer could not load the image", async () => {
    mockCreateViewer.mockRejectedValue(new ViewerUnavailableError("failed"));
    render(<StudyViewer imageUrl="http://api.test/image" />);
    expect(
      await screen.findByText("The viewer could not load the image."),
    ).toBeInTheDocument();
  });
});
