import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FedShield — Research Console",
  description: "Federated fraud-model evaluation across simulated, non-IID bank datasets.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
