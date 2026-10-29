import { render, screen } from "@testing-library/react";
import { CallToAction } from "./CallToAction";
import { Hero } from "./Hero";

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

describe("landing links to the portal", () => {
  it("opens the portal from the hero", () => {
    render(<Hero />);
    expect(
      screen.getByRole("link", { name: "Open the portal" }),
    ).toHaveAttribute("href", "/studies");
  });

  it("opens the portal from the closing call to action", () => {
    render(<CallToAction />);
    expect(
      screen.getByRole("link", { name: "Open the portal" }),
    ).toHaveAttribute("href", "/studies");
    expect(
      screen.queryByRole("button", { name: /demo/i }),
    ).not.toBeInTheDocument();
  });
});
