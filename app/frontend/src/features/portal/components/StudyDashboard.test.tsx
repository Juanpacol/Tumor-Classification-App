import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  ApiError,
  type ApiClient,
  type Page,
  type StudyDetail,
} from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { StudyDashboard } from "./StudyDashboard";

function clientWith(listStudies: ApiClient["listStudies"]): ApiClient {
  return { ...createMockClient(), listStudies };
}

function pageOf(items: StudyDetail[], extra: Partial<Page<StudyDetail>> = {}) {
  return { items, page: 1, page_size: 10, total: items.length, ...extra };
}

describe("StudyDashboard", () => {
  it("shows a loading state, then the studies newest first", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading studies");
    const rows = await screen.findAllByRole("link");
    expect(rows[0]).toHaveTextContent("Oct 28, 2026");
    expect(rows[0]).toHaveTextContent("Meningioma");
    expect(rows[0]).toHaveTextContent("93%");
  });

  it("links each row to its workspace", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    const rows = await screen.findAllByRole("link");
    expect(rows[0]).toHaveAttribute("href", "/studies/study-confident");
  });

  it("highlights the studies that need expert review", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    expect(screen.getAllByText("Needs expert review")).toHaveLength(1);
    const flagged = screen.getByText("Needs expert review").closest("a");
    expect(flagged).toHaveAttribute("href", "/studies/study-uncertain");
  });

  it("shows the status of rejected and failed studies without a prediction", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    const rejected = screen.getByText("Rejected").closest("a");
    expect(rejected).toHaveTextContent("—");
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    render(<StudyDashboard client={clientWith(async () => pageOf([]))} />);
    expect(await screen.findByText("No studies yet.")).toBeInTheDocument();
  });

  it("shows the error and retries the request", async () => {
    const listStudies = jest
      .fn<ReturnType<ApiClient["listStudies"]>, []>()
      .mockRejectedValueOnce(
        new ApiError("NETWORK_ERROR", "The server could not be reached."),
      )
      .mockResolvedValueOnce(pageOf([]));
    render(<StudyDashboard client={clientWith(listStudies)} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The server could not be reached.",
    );
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("No studies yet.")).toBeInTheDocument();
    expect(listStudies).toHaveBeenCalledTimes(2);
  });

  it("pages through the studies", async () => {
    const study = createMockClient().getStudy("study-confident");
    const listStudies = jest.fn(async ({ page = 1 } = {}) =>
      pageOf([await study], { page, total: 25 }),
    );
    render(<StudyDashboard client={clientWith(listStudies)} />);
    expect(await screen.findByText("Page 1 of 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByText("Page 2 of 3")).toBeInTheDocument();
    await waitFor(() =>
      expect(listStudies).toHaveBeenLastCalledWith({ page: 2, pageSize: 10 }),
    );
  });
});
