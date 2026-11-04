import { render } from "@testing-library/react";
import { Architecture } from "./Architecture";
import { Honesty } from "./Honesty";
import { TechStack } from "./TechStack";

beforeAll(() => {
  window.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  } as unknown as typeof IntersectionObserver;
});

function pageText(): string {
  const { container } = render(
    <>
      <TechStack />
      <Architecture />
      <Honesty />
    </>,
  );
  return container.textContent ?? "";
}

describe("How we built Ether", () => {
  it("names the language model Ether really uses and the real backend pattern", () => {
    const text = pageText();
    expect(text).toContain("Muse Spark");
    expect(text).toContain("Hexagonal");
    expect(text).not.toMatch(/on-premise/i);
  });

  it("promises no DICOM support, since uploads are JPG and PNG", () => {
    expect(pageText()).not.toMatch(/dicom/i);
  });

  it("explains the verified report and states what Ether does not claim", () => {
    const text = pageText();
    expect(text).toMatch(/not verified/i);
    expect(text).toMatch(/not a medical device/i);
  });

  it("shows no model accuracy figure until the team has reviewed the model", () => {
    expect(pageText()).not.toMatch(/f1|accuracy|\d{2}\s?%/i);
  });
});
