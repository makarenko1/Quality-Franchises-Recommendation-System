import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Movie Recommendation Survey",
  description: "Recommendation survey for the movie recommender project",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
