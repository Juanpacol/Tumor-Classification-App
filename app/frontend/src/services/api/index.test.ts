import { createApiClient } from ".";

describe("createApiClient", () => {
  it("serves sample data in mock mode", async () => {
    const client = createApiClient({ mode: "mock" });
    const page = await client.listStudies();
    expect(page.items.length).toBeGreaterThan(0);
  });

  it("defaults to the HTTP client", () => {
    const client = createApiClient({ baseUrl: "http://api.test" });
    expect(client.imageUrl("1")).toBe("http://api.test/api/studies/1/image");
  });
});
