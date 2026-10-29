import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Alert, Button, Card, StatusTag } from ".";

describe("Button", () => {
  it("calls onClick and is a non-submitting button by default", async () => {
    const onClick = jest.fn();
    render(<Button onClick={onClick}>Upload</Button>);
    const button = screen.getByRole("button", { name: "Upload" });
    expect(button).toHaveAttribute("type", "button");
    await userEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not fire when disabled", async () => {
    const onClick = jest.fn();
    render(
      <Button disabled onClick={onClick}>
        Upload
      </Button>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Upload" }));
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe("Alert", () => {
  it("announces a warning as an alert with its title", () => {
    render(
      <Alert tone="warning" title="Needs expert review">
        The model is uncertain.
      </Alert>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Needs expert review");
    expect(screen.getByRole("alert")).toHaveTextContent(
      "The model is uncertain.",
    );
  });

  it("announces a neutral alert as a status without a title", () => {
    render(<Alert>Decision support only.</Alert>);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Decision support only.",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("Card", () => {
  it("renders its content in a section", () => {
    render(<Card tone="plate">Viewer</Card>);
    expect(screen.getByText("Viewer").tagName).toBe("SECTION");
  });
});

describe("StatusTag", () => {
  it("renders the status text", () => {
    render(<StatusTag tone="attention">Triage</StatusTag>);
    expect(screen.getByText("Triage")).toBeInTheDocument();
  });
});
