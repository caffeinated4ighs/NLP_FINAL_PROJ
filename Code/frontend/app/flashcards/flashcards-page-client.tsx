"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, Brain, Loader2, AlertCircle, RotateCcw } from "lucide-react"

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
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"

type FlashcardsResponse = {
  session_id: string
  flashcards: string
}

type ParsedFlashcard = {
  front: string
  back: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

function parseFlashcards(raw: string): ParsedFlashcard[] {
  const cards: ParsedFlashcard[] = []

  const blocks = raw
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean)

  for (const block of blocks) {
    const frontMatch = block.match(/Front:\s*(.*)/i)
    const backMatch = block.match(/Back:\s*([\s\S]*)/i)

    if (frontMatch && backMatch) {
      cards.push({
        front: frontMatch[1].trim(),
        back: backMatch[1].trim(),
      })
    }
  }

  if (cards.length > 0) return cards

  return [
    {
      front: "Generated flashcards",
      back: raw,
    },
  ]
}

export default function FlashcardsPageClient() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [sessionId, setSessionId] = useState("")
  const [numCards, setNumCards] = useState(4)
  const [rawFlashcards, setRawFlashcards] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState("")

  const parsedCards = useMemo(
    () => parseFlashcards(rawFlashcards),
    [rawFlashcards]
  )

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
    generateFlashcards(sessionId, numCards)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  async function generateFlashcards(activeSessionId = sessionId, count = numCards) {
    if (!activeSessionId) {
      setError("Missing session_id.")
      return
    }

    setIsGenerating(true)
    setError("")
    setRawFlashcards("")

    try {
      const response = await fetch(`${API_BASE}/api/flashcards`, {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          num_cards: count,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Flashcards request failed.")
      }

      const data: FlashcardsResponse = await response.json()

      localStorage.setItem("rag_flashcards", data.flashcards)
      setRawFlashcards(data.flashcards)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown flashcards error.")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <main className="min-h-screen bg-muted/40 px-6 py-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <div className="flex items-center justify-between gap-4">
          <Button variant="outline" onClick={() => router.push(`/status?session_id=${sessionId}`)}>
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
                <Brain className="h-6 w-6 text-primary" />
              </div>

              <div>
                <CardTitle className="text-2xl">Flashcards</CardTitle>
                <CardDescription>
                  Generate study flashcards from the indexed coursework.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="space-y-2">
                <label className="text-sm font-medium">Number of cards</label>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={numCards}
                  onChange={(e) => setNumCards(Number(e.target.value))}
                  className="w-40"
                />
              </div>

              <Button
                onClick={() => generateFlashcards(sessionId, numCards)}
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
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Flashcards failed</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {isGenerating && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertTitle>Generating flashcards</AlertTitle>
                <AlertDescription>
                  Calling <code>POST /api/flashcards</code> with this session.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {rawFlashcards && (
          <div className="grid gap-4 md:grid-cols-2">
            {parsedCards.map((card, index) => (
              <Card key={`${card.front}-${index}`} className="overflow-hidden">
                <CardHeader>
                  <div className="flex items-center justify-between gap-3">
                    <CardTitle className="text-lg">Card {index + 1}</CardTitle>
                    <Badge variant="outline">Flashcard</Badge>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  <div>
                    <div className="mb-1 text-xs font-semibold uppercase text-muted-foreground">
                      Front
                    </div>
                    <div className="rounded-lg border bg-background p-3 text-sm font-medium">
                      {card.front}
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <div className="mb-1 text-xs font-semibold uppercase text-muted-foreground">
                      Back
                    </div>
                    <div className="rounded-lg bg-muted p-3 text-sm leading-6">
                      {card.back}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {rawFlashcards && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Raw backend response</CardTitle>
              <CardDescription>
                Useful for debugging parser behavior.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="whitespace-pre-wrap rounded-lg bg-zinc-950 p-4 text-sm text-zinc-50">
                {rawFlashcards}
              </pre>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}