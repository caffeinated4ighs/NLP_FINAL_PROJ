import type { Metadata } from "next"
import "katex/dist/katex.min.css"
import "./globals.css"

export const metadata: Metadata = {
  title: "Coursework RAG Tutor",
  description: "Multimodal coursework RAG tutor",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}