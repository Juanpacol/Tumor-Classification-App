import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError, type ApiClient } from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { UploadStudy } from "./UploadStudy";

function clientWith(uploadStudy: ApiClient["uploadStudy"]): ApiClient {
  return { ...createMockClient(), uploadStudy };
}

const file = new File(["mri"], "scan.png", { type: "image/png" });

describe("UploadStudy", () => {
  it("disables the submit button until a file is picked", () => {
    render(<UploadStudy client={createMockClient()} onUploaded={jest.fn()} />);
    expect(screen.getByRole("button", { name: "Upload study" })).toBeDisabled();
  });

  it("uploads the picked file with trimmed notes and opens the new study", async () => {
    const uploadStudy = jest
      .fn()
      .mockResolvedValue({ id: "study-new", prediction: null, report: null });
    const onUploaded = jest.fn();
    render(
      <UploadStudy client={clientWith(uploadStudy)} onUploaded={onUploaded} />,
    );

    await userEvent.upload(screen.getByLabelText("MRI image"), file);
    await userEvent.type(
      screen.getByLabelText("Clinical notes (optional)"),
      "  54F, headaches  ",
    );
    await userEvent.click(screen.getByRole("button", { name: "Upload study" }));

    expect(uploadStudy).toHaveBeenCalledWith(file, "54F, headaches");
    expect(onUploaded).toHaveBeenCalledWith("study-new");
  });

  it("shows the error when the upload fails", async () => {
    const uploadStudy = jest
      .fn()
      .mockRejectedValue(new ApiError("UNSUPPORTED_IMAGE", "Not an MRI."));
    render(
      <UploadStudy client={clientWith(uploadStudy)} onUploaded={jest.fn()} />,
    );

    await userEvent.upload(screen.getByLabelText("MRI image"), file);
    await userEvent.click(screen.getByRole("button", { name: "Upload study" }));

    expect(
      await screen.findByText("The study could not be uploaded"),
    ).toBeInTheDocument();
    expect(screen.getByText("Not an MRI.")).toBeInTheDocument();
  });
});
