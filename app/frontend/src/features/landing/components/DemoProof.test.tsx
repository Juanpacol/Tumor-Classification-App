import { render, screen } from "@testing-library/react";
import { DemoProof } from "./DemoProof";

beforeAll(() => {
  global.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  } as unknown as typeof IntersectionObserver;
});

function mockReducedMotion(matches: boolean) {
  window.matchMedia = jest.fn().mockReturnValue({ matches }) as never;
}

describe("DemoProof", () => {
  it("plays a silent looping clip with a poster", () => {
    mockReducedMotion(false);
    const { container } = render(<DemoProof />);
    const video = container.querySelector("video") as HTMLVideoElement;

    expect(video).toHaveAttribute("src", "/demo/ether-demo-loop.mp4");
    expect(video).toHaveAttribute("poster", "/demo/ether-demo-poster.jpg");
    expect(video.muted).toBe(true);
    expect(video.loop).toBe(true);
    expect(video).not.toHaveAttribute("controls");
  });

  it("says it is sample data and not a medical device", () => {
    mockReducedMotion(false);
    render(<DemoProof />);

    expect(
      screen.getByText(/Sample data from a public dataset/),
    ).toHaveTextContent("not a medical device");
  });

  it("links to the full video", () => {
    mockReducedMotion(false);
    render(<DemoProof />);

    expect(screen.getByRole("link", { name: /full video/i })).toHaveAttribute(
      "href",
      "https://www.youtube.com/watch?v=rXtExvxENqg",
    );
  });

  it("does not autoplay and shows controls with reduced motion", () => {
    mockReducedMotion(true);
    window.HTMLMediaElement.prototype.pause = jest.fn();
    const { container } = render(<DemoProof />);
    const video = container.querySelector("video") as HTMLVideoElement;

    expect(video).toHaveAttribute("controls");
    expect(video.autoplay).toBe(false);
  });
});
