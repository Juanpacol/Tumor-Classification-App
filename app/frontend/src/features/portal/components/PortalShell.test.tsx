import { render, screen } from "@testing-library/react";
import { DISCLAIMER } from "./PortalHeader";
import { PortalShell } from "./PortalShell";

describe("PortalShell", () => {
  it("always shows the decision-support disclaimer with the page", () => {
    render(
      <PortalShell>
        <p>Page content</p>
      </PortalShell>,
    );
    expect(screen.getByRole("note")).toHaveTextContent(DISCLAIMER);
    expect(screen.getByText("Page content")).toBeInTheDocument();
  });
});
