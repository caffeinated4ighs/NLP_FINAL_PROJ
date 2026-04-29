"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  ArrowLeft,
  AlertCircle,
  HelpCircle,
  Loader2,
  RotateCcw,
  Eye,
  EyeOff,
} from "lucide-react"

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
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"

type QuizResponse = {
  session_id: string
  quiz: string
}

type ParsedQuestion = {
  number: string
  question: string
  type: string
  choices: string[]
  answer: string
  source: string
  raw: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

function cleanMarkdownLabel(value: string) {
  return value
    .replace(/\*\*/g, "")
    .replace(/^[-*]\s*/, "")
    .trim()
}

function parseQuiz(raw: string): ParsedQuestion[] {
  const withoutTitle = raw.replace(/^##\s*Quiz\s*/i, "").trim()

  const blocks = withoutTitle
    .split(/\n(?=\d+\.\s+\*\*Question:\*\*)/g)
    .map((block) => block.trim())
    .filter(Boolean)

  const parsed = blocks.map((block) => {
    const numberMatch = block.match(/^(\d+)\.\s+/)

    const questionMatch = block.match(
      /\*\*Question:\*\*\s*([\s\S]*?)(?=\n\s*\*\*Type:\*\*|\n\s*-\s*[A-D]\)|\n\s*\*\*Answer:\*\*|$)/i
    )

    const typeMatch = block.match(/\*\*Type:\*\*\s*(.*)/i)

    const answerMatch = block.match(
      /\*\*Answer:\*\*\s*([\s\S]*?)(?=\n\s*\*\*Source reference:\*\*|$)/i
    )

    const sourceMatch = block.match(/\*\*Source reference:\*\*\s*([\s\S]*)/i)

    const choices = Array.from(
      block.matchAll(/^\s*-\s*([A-D]\)[\s\S]*?)$/gim)
    ).map((match) => cleanMarkdownLabel(match[1]))

    return {
      number: numberMatch?.[1] || "",
      question: cleanMarkdownLabel(questionMatch?.[1] || "Question not parsed."),
      type: cleanMarkdownLabel(typeMatch?.[1] || "Unknown"),
      choices,
      answer: cleanMarkdownLabel(answerMatch?.[1] || "Answer not parsed."),
      source: cleanMarkdownLabel(sourceMatch?.[1] || ""),
      raw: block,
    }
  })

  return parsed.length > 0
    ? parsed
    : [
        {
          number: "1",
          question: "Quiz output",
          type: "Raw markdown",
          choices: [],
          answer: raw,
          source: "",
          raw,
        },
      ]
}

export default function QuizPageClient() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [sessionId, setSessionId] = useState("")
  const [numQuestions, setNumQuestions] = useState(10)
  const [difficulty, setDifficulty] = useState("mixed")
  const [rawQuiz, setRawQuiz] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState("")
  const [visibleAnswers, setVisibleAnswers] = useState<Record<number, boolean>>({})

  const questions = useMemo(() => parseQuiz(rawQuiz), [rawQuiz])

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
    generateQuiz(sessionId, numQuestions, difficulty)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  async function generateQuiz(
    activeSessionId = sessionId,
    count = numQuestions,
    level = difficulty
  ) {
    if (!activeSessionId) {
      setError("Missing session_id.")
      return
    }

    setIsGenerating(true)
    setError("")
    setRawQuiz("")
    setVisibleAnswers({})

    try {
      const response = await fetch(`${API_BASE}/api/quiz`, {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          num_questions: count,
          difficulty: level,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Quiz request failed.")
      }

      const data: QuizResponse = await response.json()

      localStorage.setItem("rag_quiz", data.quiz)
      setRawQuiz(data.quiz)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown quiz error.")
    } finally {
      setIsGenerating(false)
    }
  }

  function toggleAnswer(index: number) {
    setVisibleAnswers((prev) => ({
      ...prev,
      [index]: !prev[index],
    }))
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
                <HelpCircle className="h-6 w-6 text-primary" />
              </div>

              <div>
                <CardTitle className="text-2xl">Quiz</CardTitle>
                <CardDescription>
                  Generates a markdown quiz and displays each question as a card.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="space-y-2">
                <label className="text-sm font-medium">Questions</label>
                <Input
                  type="number"
                  min={1}
                  max={25}
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(Number(e.target.value))}
                  className="w-36"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Difficulty</label>
                <Input
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  className="w-40"
                  placeholder="mixed"
                />
              </div>

              <Button
                onClick={() => generateQuiz(sessionId, numQuestions, difficulty)}
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
                <AlertTitle>Quiz failed</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {isGenerating && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertTitle>Generating quiz</AlertTitle>
                <AlertDescription>
                  Calling <code>POST /api/quiz</code>.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {rawQuiz && (
          <div className="space-y-4">
            {questions.map((item, index) => (
              <Card key={`${item.number}-${index}`}>
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <CardTitle className="text-lg">
                        Question {item.number || index + 1}
                      </CardTitle>
                      <CardDescription>{item.type}</CardDescription>
                    </div>

                    <Badge variant="outline">{item.type}</Badge>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  <div className="rounded-lg border bg-background p-4">
                    <div className="prose prose-zinc max-w-none dark:prose-invert">
                      <MarkdownMath>{item.question}</MarkdownMath>
                    </div>
                  </div>

                  {item.choices.length > 0 && (
                    <div className="space-y-2">
                      {item.choices.map((choice) => (
                        <div
                          key={choice}
                          className="rounded-lg border bg-muted/40 p-3 text-sm"
                        >
                          <div className="prose prose-zinc max-w-none dark:prose-invert">
                            <MarkdownMath>{choice}</MarkdownMath>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <Button
                    variant="secondary"
                    onClick={() => toggleAnswer(index)}
                  >
                    {visibleAnswers[index] ? (
                      <>
                        <EyeOff className="mr-2 h-4 w-4" />
                        Hide Answer
                      </>
                    ) : (
                      <>
                        <Eye className="mr-2 h-4 w-4" />
                        Show Answer
                      </>
                    )}
                  </Button>

                  {visibleAnswers[index] && (
                    <div className="rounded-lg border border-green-200 bg-green-50 p-4 text-sm text-green-950">
                      <div className="mb-2 font-semibold">Answer</div>
                      <div className="prose prose-zinc max-w-none">
                        <MarkdownMath>{item.answer}</MarkdownMath>
                      </div>
                    </div>
                  )}

                  {item.source && (
                    <>
                      <Separator />
                      <details className="text-xs text-muted-foreground">
                        <summary className="cursor-pointer">Source reference</summary>
                        <div className="mt-2 break-all rounded-md bg-muted p-3">
                          {item.source}
                        </div>
                      </details>
                    </>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {rawQuiz && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Raw backend response</CardTitle>
              <CardDescription>
                Useful for debugging parser behavior.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-invert max-w-none rounded-lg bg-zinc-950 p-4 text-sm text-zinc-50">
                <MarkdownMath>{rawQuiz}</MarkdownMath>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}