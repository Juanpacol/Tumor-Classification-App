import { createViewer } from "./viewerEngine";

function answerWith(response: { ok: boolean; status: number } | Error) {
  global.fetch = jest.fn(async () => {
    if (response instanceof Error) throw response;
    return response;
  }) as unknown as typeof fetch;
}

describe("createViewer", () => {
  beforeEach(() => jest.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => jest.restoreAllMocks());

  it("reports a missing image whatever the rest of the stack does", async () => {
    answerWith({ ok: false, status: 404 });
    await expect(
      createViewer(document.createElement("div"), "http://api.test/image"),
    ).rejects.toMatchObject({ reason: "image_not_found" });
  });

  it("reports any other failure to download the image", async () => {
    answerWith({ ok: false, status: 500 });
    await expect(
      createViewer(document.createElement("div"), "http://api.test/image"),
    ).rejects.toMatchObject({ reason: "failed" });
  });

  it("reports an image that could not be reached", async () => {
    answerWith(new TypeError("fetch failed"));
    await expect(
      createViewer(document.createElement("div"), "http://api.test/image"),
    ).rejects.toMatchObject({ reason: "failed" });
  });

  it("logs the cause of the failure", async () => {
    answerWith({ ok: false, status: 404 });
    await createViewer(
      document.createElement("div"),
      "http://api.test/x",
    ).catch(() => undefined);
    expect(console.error).toHaveBeenCalledWith(
      "The viewer could not show the image.",
      expect.objectContaining({ status: 404 }),
    );
  });
});
