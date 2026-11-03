import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { SmoothScrollProvider } from "./SmoothScrollProvider";

const barlowCondensed = localFont({
  src: [
    {
      path: "../../node_modules/@fontsource/barlow-condensed/files/barlow-condensed-latin-300-normal.woff2",
      weight: "300",
      style: "normal",
    },
    {
      path: "../../node_modules/@fontsource/barlow-condensed/files/barlow-condensed-latin-500-normal.woff2",
      weight: "500",
      style: "normal",
    },
    {
      path: "../../node_modules/@fontsource/barlow-condensed/files/barlow-condensed-latin-600-normal.woff2",
      weight: "600",
      style: "normal",
    },
  ],
  variable: "--font-barlow-condensed",
});

const inter = localFont({
  src: "../../node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2",
  weight: "100 900",
  variable: "--font-inter",
});

const jetbrainsMono = localFont({
  src: "../../node_modules/@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2",
  weight: "100 800",
  variable: "--font-jetbrains-mono",
});

const playfairDisplay = localFont({
  src: [
    {
      path: "../../node_modules/@fontsource-variable/playfair-display/files/playfair-display-latin-wght-normal.woff2",
      weight: "400 900",
      style: "normal",
    },
    {
      path: "../../node_modules/@fontsource-variable/playfair-display/files/playfair-display-latin-wght-italic.woff2",
      weight: "400 900",
      style: "italic",
    },
  ],
  variable: "--font-playfair-display",
});

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
    <html
      lang="en"
      className={`${barlowCondensed.variable} ${inter.variable} ${jetbrainsMono.variable} ${playfairDisplay.variable}`}
    >
      <body>
        <SmoothScrollProvider />
        {children}
      </body>
    </html>
  );
}
