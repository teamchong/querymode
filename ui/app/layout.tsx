import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "QueryMode Studio",
  description: "Data explorer for QueryMode",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <style dangerouslySetInnerHTML={{ __html: `
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body { font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; background: #0d1117; color: #c9d1d9; }
          ::-webkit-scrollbar { width: 6px; height: 6px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        `}} />
      </head>
      <body>{children}</body>
    </html>
  );
}
