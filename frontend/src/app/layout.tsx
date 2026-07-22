import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { QueryProvider } from "@/components/query-provider";
import "./globals.css";
import "@/components/ui/styles.css";

export const metadata: Metadata = {
  title: "PROJECT-KAGYA",
  description: "Subjective AI runtime interface",
};

const navItems = [
  ["Chat", "/chat"],
  ["Outbox", "/outbox"],
  ["Debug", "/debug"],
  ["Memory", "/memory"],
  ["Sleep", "/sleep"],
  ["Datasets", "/datasets"],
  ["Adapters", "/adapters"],
  ["Evaluations", "/evaluations"],
];

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <QueryProvider>
          <div className="app-shell">
            <aside className="sidebar">
              <div className="brand">PROJECT-KAGYA</div>
              <nav className="nav" aria-label="Primary navigation">
                {navItems.map(([label, href]) => (
                  <Link key={href} href={href}>{label}</Link>
                ))}
              </nav>
            </aside>
            <main className="main">{children}</main>
          </div>
        </QueryProvider>
      </body>
    </html>
  );
}
