import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Ether",
  description: "Hybrid Brain Tumor Detection Support System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
