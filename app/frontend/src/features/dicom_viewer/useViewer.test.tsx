import {
  act,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { useViewer } from "./useViewer";
import {
  ViewerUnavailableError,
  type CreateViewer,
  type Viewer,
} from "./viewerEngine";

function fakeViewer(): Viewer {
  return { setTool: jest.fn(), reset: jest.fn(), destroy: jest.fn() };
}

function Harness({ url, create }: { url: string; create: CreateViewer }) {
  const { containerRef, state } = useViewer(url, create);
  return (
    <div>
      <div ref={containerRef} />
      <span>
        {state.status === "unavailable" ? state.reason : state.status}
      </span>
    </div>
  );
}

describe("useViewer", () => {
  it("becomes ready once the viewer is created", async () => {
    const create: CreateViewer = jest.fn(async () => fakeViewer());
    render(<Harness url="a" create={create} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    expect(await screen.findByText("ready")).toBeInTheDocument();
  });

  it.each(["no_webgl", "image_not_found"] as const)(
    "reports %s as unavailable",
    async (reason) => {
      const create: CreateViewer = jest.fn(async () => {
        throw new ViewerUnavailableError(reason);
      });
      render(<Harness url="a" create={create} />);
      expect(await screen.findByText(reason)).toBeInTheDocument();
    },
  );

  it("reports an unexpected failure as failed", async () => {
    const create: CreateViewer = jest.fn(async () => {
      throw new Error("boom");
    });
    render(<Harness url="a" create={create} />);
    expect(await screen.findByText("failed")).toBeInTheDocument();
  });

  it("destroys the viewer on unmount", async () => {
    const viewer = fakeViewer();
    const { unmount } = render(<Harness url="a" create={async () => viewer} />);
    await screen.findByText("ready");
    unmount();
    expect(viewer.destroy).toHaveBeenCalledTimes(1);
  });

  it("destroys a viewer that finishes loading after the unmount", async () => {
    const viewer = fakeViewer();
    let finish: (viewer: Viewer) => void = () => {};
    const create: CreateViewer = () =>
      new Promise<Viewer>((resolve) => {
        finish = resolve;
      });
    const { unmount } = render(<Harness url="a" create={create} />);
    unmount();
    await act(async () => finish(viewer));
    expect(viewer.destroy).toHaveBeenCalledTimes(1);
  });

  it("releases every viewer when moving across ten studies", async () => {
    const viewers: Viewer[] = [];
    const create: CreateViewer = jest.fn(async () => {
      const viewer = fakeViewer();
      viewers.push(viewer);
      return viewer;
    });
    const { rerender, unmount } = render(<Harness url="0" create={create} />);
    for (let study = 1; study < 10; study += 1) {
      await waitFor(() => expect(viewers).toHaveLength(study));
      rerender(<Harness url={String(study)} create={create} />);
    }
    await waitFor(() => expect(viewers).toHaveLength(10));
    unmount();
    viewers.forEach((viewer) =>
      expect(viewer.destroy).toHaveBeenCalledTimes(1),
    );
  });

  it("forwards the tool and the reset to the current viewer", async () => {
    const viewer = fakeViewer();
    const create: CreateViewer = async () => viewer;
    let controls: ReturnType<typeof useViewer> | undefined;
    function Probe() {
      controls = useViewer("a", create);
      return <div ref={controls.containerRef} />;
    }
    render(<Probe />);
    await waitFor(() => expect(controls?.state.status).toBe("ready"));
    controls?.setTool("zoom");
    controls?.reset();
    expect(viewer.setTool).toHaveBeenCalledWith("zoom");
    expect(viewer.reset).toHaveBeenCalled();
  });
});
