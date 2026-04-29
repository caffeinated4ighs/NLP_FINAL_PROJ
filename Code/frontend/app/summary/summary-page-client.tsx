"use client"

import { useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, AlertCircle, FileText, Loader2, RotateCcw } from "lucide-react"

import { MarkdownMath } from "@/components/markdown-math"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

type SummaryResponse = {
  session_id: string
  summary: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export default function SummaryPageClient() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [sessionId, setSessionId] = useState("")
  const [summary, setSummary] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    const fromUrl = searchParams.get("session_id")
    const fromStorage =
      typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : ""

    const activeSessionId = fromUrl || fromStorage || ""

    if (!activeSessionId) {
      setError("No session_id found. Go back and upload files first.")
      return
    }

    setSessionId(activeSessionId)
  }, [searchParams])

  useEffect(() => {
    if (!sessionId) return
    generateSummary(sessionId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  async function generateSummary(activeSessionId = sessionId) {
    if (!activeSessionId) {
      setError("Missing session_id.")
      return
    }

    setIsGenerating(true)
    setError("")
    setSummary("")

    try {
      const response = await fetch(`${API_BASE}/api/summary`, {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: activeSessionId,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Summary request failed.")
      }

      const data: SummaryResponse = await response.json()

      localStorage.setItem("rag_summary", data.summary)
      setSummary(data.summary)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown summary error.")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <main className="min-h-screen bg-muted/40 px-6 py-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-center justify-between gap-4">
          <Button
            variant="outline"
            onClick={() => router.push(`/status?session_id=${sessionId}`)}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to status
          </Button>

          <Badge variant="secondary" className="max-w-[420px] truncate">
            {sessionId || "No session"}
          </Badge>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-primary/10 p-3">
                <FileText className="h-6 w-6 text-primary" />
              </div>

              <div>
                <CardTitle className="text-2xl">Coursework Summary</CardTitle>
                <CardDescription>
                  Generates a structured markdown summary from the indexed material.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-5">
            <Button
              onClick={() => generateSummary(sessionId)}
              disabled={isGenerating || !sessionId}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <RotateCcw className="mr-2 h-4 w-4" />
                  Generate Again
                </>
              )}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Summary failed</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {isGenerating && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertTitle>Generating summary</AlertTitle>
                <AlertDescription>
                  Calling <code>POST /api/summary</code>.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {summary && (
          <Card>
            <CardContent className="pt-6">
              <div className="prose prose-zinc max-w-none dark:prose-invert">
                <MarkdownMath>{summary}</MarkdownMath>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}