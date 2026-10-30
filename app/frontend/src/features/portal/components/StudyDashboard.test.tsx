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

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: jest.fn() }),
  usePathname: () => "/studies",
}));

function clientWith(listStudies: ApiClient["listStudies"]): ApiClient {
  return { ...createMockClient(), listStudies };
}

function pageOf(items: StudyDetail[], extra: Partial<Page<StudyDetail>> = {}) {
  return { items, page: 1, page_size: 10, total: items.length, ...extra };
}

describe("StudyDashboard", () => {
  it("shows a loading state, then the studies by priority", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading studies");
    const rows = await screen.findAllByRole("link");
    expect(rows[0]).toHaveTextContent("Glioma");
    expect(rows[0]).toHaveTextContent("52%");
    expect(rows[0]).toHaveTextContent("Needs expert review");
  });

  it("puts the studies that need a reader before the finished ones", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    const hrefs = (await screen.findAllByRole("link")).map((row) =>
      row.getAttribute("href"),
    );
    expect(hrefs.indexOf("/studies/study-uncertain")).toBe(0);
    expect(hrefs.indexOf("/studies/study-rejected")).toBeLessThan(
      hrefs.indexOf("/studies/study-confident"),
    );
  });

  it("links each row to its workspace", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    const rows = await screen.findAllByRole("link");
    expect(rows[0]).toHaveAttribute("href", "/studies/study-uncertain");
  });

  it("keeps the facts a reader would open the study for inside each row", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    expect(
      screen.getByText("The image is too noisy to classify."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Meningioma 93% · Glioma 2% · Pituitary 2%"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("The analysis is still running."),
    ).toBeInTheDocument();
  });

  it("filters by category from the summary cards", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    await userEvent.click(
      screen.getByRole("button", { name: /Need expert review/ }),
    );
    expect(screen.getAllByRole("link")).toHaveLength(1);
    await userEvent.click(
      screen.getByRole("button", { name: /Need expert review/ }),
    );
    expect(screen.getAllByRole("link").length).toBeGreaterThan(1);
  });

  it("searches by class", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    await userEvent.type(screen.getByLabelText("Search studies"), "meni");
    const rows = screen.getAllByRole("link");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveAttribute("href", "/studies/study-confident");
  });

  it("says when nothing matches the search", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    await userEvent.type(screen.getByLabelText("Search studies"), "zzz");
    expect(
      screen.getByText("No studies match this filter."),
    ).toBeInTheDocument();
  });

  it("can sort the newest first", async () => {
    render(<StudyDashboard client={createMockClient()} />);
    await screen.findAllByRole("link");
    await userEvent.click(screen.getByRole("button", { name: "Newest" }));
    expect(screen.getAllByRole("link")[0]).toHaveAttribute(
      "href",
      "/studies/study-confident",
    );
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
    expect(screen.getByRole("button", { name: /Previous/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("Page 2 of 3")).toBeInTheDocument();
    await waitFor(() =>
      expect(listStudies).toHaveBeenLastCalledWith({ page: 2, pageSize: 10 }),
    );
  });
});
