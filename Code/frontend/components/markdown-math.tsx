"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"

function normalizeMathMarkdown(text: string) {
  return text
    // Convert \( ... \) to $ ... $
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expr) => `$${expr}$`)

    // Convert \[ ... \] to $$ ... $$
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expr) => `$$\n${expr}\n$$`)

    // Fix common escaped LaTeX commands from model/JSON output.
    .replace(/\\\\frac/g, "\\frac")
    .replace(/\\\\cap/g, "\\cap")
    .replace(/\\\\cup/g, "\\cup")
    .replace(/\\\\leq/g, "\\leq")
    .replace(/\\\\geq/g, "\\geq")
    .replace(/\\\\hat/g, "\\hat")
    .replace(/\\\\theta/g, "\\theta")
    .replace(/\\\\sum/g, "\\sum")
    .replace(/\\\\infty/g, "\\infty")
    .replace(/\\\\to/g, "\\to")
}

export function MarkdownMath({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
    >
      {normalizeMathMarkdown(children)}
    </ReactMarkdown>
  )
}