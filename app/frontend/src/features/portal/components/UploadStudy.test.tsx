import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError, type ApiClient } from "@/services/api";
import { createMockClient } from "@/services/api/mockClient";
import { MAX_UPLOAD_BYTES } from "../uploadChecks";
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

  it("opens a study the classifier rejected like any other", async () => {
    const uploadStudy = jest.fn().mockResolvedValue({
      id: "study-rejected",
      status: "rejected",
      rejection_reason: "too noisy",
      prediction: null,
      report: null,
    });
    const onUploaded = jest.fn();
    render(
      <UploadStudy client={clientWith(uploadStudy)} onUploaded={onUploaded} />,
    );

    await userEvent.upload(screen.getByLabelText("MRI image"), file);
    await userEvent.click(screen.getByRole("button", { name: "Upload study" }));

    expect(onUploaded).toHaveBeenCalledWith("study-rejected");
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
  it("rejects a file that is not a JPG or PNG before sending it", async () => {
    const uploadStudy = jest.fn();
    render(
      <UploadStudy client={clientWith(uploadStudy)} onUploaded={jest.fn()} />,
    );

    await userEvent.upload(
      screen.getByLabelText("MRI image"),
      new File(["dicom"], "scan.dcm"),
      { applyAccept: false },
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Use a JPG or PNG");
    expect(screen.getByRole("button", { name: "Upload study" })).toBeDisabled();
  });

  it("rejects an image over 20 MB before sending it", async () => {
    render(<UploadStudy client={createMockClient()} onUploaded={jest.fn()} />);
    const large = new File(["x"], "scan.png", { type: "image/png" });
    Object.defineProperty(large, "size", { value: MAX_UPLOAD_BYTES + 1 });

    await userEvent.upload(screen.getByLabelText("MRI image"), large);

    expect(screen.getByRole("alert")).toHaveTextContent("larger than 20 MB");
    expect(screen.getByRole("button", { name: "Upload study" })).toBeDisabled();
  });

  it("accepts a dropped image", () => {
    render(<UploadStudy client={createMockClient()} onUploaded={jest.fn()} />);

    fireEvent.drop(screen.getByText("MRI image"), {
      dataTransfer: { files: [file] },
    });

    expect(screen.getByText("scan.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload study" })).toBeEnabled();
  });

  it("counts the notes and warns against patient identifiers", async () => {
    render(<UploadStudy client={createMockClient()} onUploaded={jest.fn()} />);

    await userEvent.type(
      screen.getByLabelText("Clinical notes (optional)"),
      "54F",
    );

    expect(screen.getByText(/patient identifiers/)).toHaveTextContent("3/2000");
  });

  it.each([
    [413, "larger than 20 MB"],
    [415, "could not read this file"],
    [422, "notes are longer than 2000 characters"],
    [500, "failed while processing"],
    [503, "not available right now"],
  ])(
    "explains a %i answer and keeps the file for a retry",
    async (status, text) => {
      const uploadStudy = jest
        .fn()
        .mockRejectedValue(new ApiError("ANY", "raw", status));
      render(
        <UploadStudy client={clientWith(uploadStudy)} onUploaded={jest.fn()} />,
      );

      await userEvent.upload(screen.getByLabelText("MRI image"), file);
      await userEvent.click(
        screen.getByRole("button", { name: "Upload study" }),
      );

      expect(await screen.findByText(new RegExp(text))).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Upload study" }),
      ).toBeEnabled();
    },
  );
});
