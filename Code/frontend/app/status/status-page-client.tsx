"use client"

import { useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  CheckCircle2,
  AlertCircle,
  Loader2,
  FileText,
  Database,
  RefreshCw,
  Brain,
  HelpCircle,
  FileTextIcon,
  ExternalLink,
  Send,
  MessageSquare,
} from "lucide-react"

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
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"

type IndexedFile = {
  path: string
  name: string
  extension: string
  file_type: string
  chunk_count: number
}

type StatusResponse = {
  session_id: string
  status: string
  ready: boolean
  uploaded_files: string[]
  indexed_files: IndexedFile[]
  error: string | null
}

type AskResponse = {
  session_id: string
  answer: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export default function StatusPageClient() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [sessionId, setSessionId] = useState("")
  const [statusData, setStatusData] = useState<StatusResponse | null>(null)
  const [error, setError] = useState("")

  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [isAsking, setIsAsking] = useState(false)
  const [askError, setAskError] = useState("")

  useEffect(() => {
    const fromUrl = searchParams.get("session_id")
    const fromStorage =
      typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : ""

    const activeSessionId = fromUrl || fromStorage || ""

    if (!activeSessionId) {
      setError("No session_id found in URL or localStorage.")
      return
    }

    setSessionId(activeSessionId)
  }, [searchParams])

  useEffect(() => {
    if (!sessionId) return

    let timer: ReturnType<typeof setInterval> | null = null

    async function fetchStatus() {
      try {
        const response = await fetch(`${API_BASE}/api/status/${sessionId}`, {
          method: "GET",
          headers: {
            accept: "application/json",
          },
        })

        if (!response.ok) {
          const message = await response.text()
          throw new Error(message || "Status request failed.")
        }

        const data: StatusResponse = await response.json()

        setStatusData(data)
        setError("")

        if (data.ready || data.status === "error") {
          if (timer) clearInterval(timer)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown status error.")
        if (timer) clearInterval(timer)
      }
    }

    fetchStatus()
    timer = setInterval(fetchStatus, 3000)

    return () => {
      if (timer) clearInterval(timer)
    }
  }, [sessionId])

  function goBackToUpload() {
    router.push("/")
  }

  function openToolInNewTab(path: string) {
    if (!sessionId) return

    window.open(
      `${path}?session_id=${sessionId}`,
      "_blank",
      "noopener,noreferrer"
    )
  }

  async function askRag() {
    if (!sessionId) {
      setAskError("Missing session_id.")
      return
    }

    if (!question.trim()) {
      setAskError("Enter a question first.")
      return
    }

    setIsAsking(true)
    setAskError("")
    setAnswer("")

    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          question: question.trim(),
          top_k: 12,
          source_contains: null,
          modality: null,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Ask request failed.")
      }

      const data: AskResponse = await response.json()
      setAnswer(data.answer)
    } catch (err) {
      setAskError(err instanceof Error ? err.message : "Unknown ask error.")
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main className="min-h-screen bg-muted/40 px-6 py-10">
      <div className="mx-auto max-w-4xl space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle className="text-2xl">Indexing Status</CardTitle>
                <CardDescription>
                  Polling backend status for the uploaded coursework session.
                </CardDescription>
              </div>

              {statusData && (
                <Badge variant={statusData.ready ? "default" : "secondary"}>
                  {statusData.status}
                </Badge>
              )}
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            {sessionId && (
              <div className="rounded-lg border bg-background p-4">
                <div className="text-sm text-muted-foreground">Session ID</div>
                <code className="break-all text-sm">{sessionId}</code>
              </div>
            )}

            {!statusData && !error && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <AlertTitle>Checking status</AlertTitle>
                <AlertDescription>
                  Calling <code>GET /api/status/{sessionId}</code>
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Status request failed</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {statusData && statusData.ready && (
              <Alert>
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Ready</AlertTitle>
                <AlertDescription>
                  Files are indexed. You can now query the coursework or open summary, quiz, and flashcards.
                </AlertDescription>
              </Alert>
            )}

            {statusData && !statusData.ready && statusData.status !== "error" && (
              <Alert>
                <RefreshCw className="h-4 w-4" />
                <AlertTitle>Indexing in progress</AlertTitle>
                <AlertDescription>
                  Backend status is <strong>{statusData.status}</strong>. This page polls every
                  3 seconds.
                </AlertDescription>
              </Alert>
            )}

            {statusData?.error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Backend error</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {statusData.error}
                </AlertDescription>
              </Alert>
            )}

            {statusData && (
              <>
                <Separator />

                <div className="grid gap-4 md:grid-cols-3">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Status</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-semibold">
                        {statusData.status}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Uploaded</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-semibold">
                        {statusData.uploaded_files.length}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Indexed</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-semibold">
                        {statusData.indexed_files.length}
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <div className="space-y-3">
                  <h3 className="flex items-center gap-2 font-medium">
                    <FileText className="h-4 w-4" />
                    Uploaded files
                  </h3>

                  <div className="rounded-lg border bg-background">
                    {statusData.uploaded_files.length === 0 && (
                      <div className="p-4 text-sm text-muted-foreground">
                        No uploaded files found.
                      </div>
                    )}

                    {statusData.uploaded_files.map((filePath, index) => (
                      <div key={filePath}>
                        <div className="break-all p-3 text-sm">{filePath}</div>
                        {index < statusData.uploaded_files.length - 1 && (
                          <Separator />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="flex items-center gap-2 font-medium">
                    <Database className="h-4 w-4" />
                    Indexed files
                  </h3>

                  <div className="rounded-lg border bg-background">
                    {statusData.indexed_files.length === 0 && (
                      <div className="p-4 text-sm text-muted-foreground">
                        No indexed files yet.
                      </div>
                    )}

                    {statusData.indexed_files.map((file, index) => (
                      <div key={file.path}>
                        <div className="grid gap-2 p-3 text-sm md:grid-cols-5">
                          <div className="font-medium md:col-span-2">
                            {file.name}
                          </div>
                          <div>{file.extension}</div>
                          <div>{file.file_type}</div>
                          <div>{file.chunk_count} chunks</div>
                        </div>

                        {index < statusData.indexed_files.length - 1 && (
                          <Separator />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button onClick={goBackToUpload} variant="outline">
                    Upload another batch
                  </Button>

                  <Button
                    variant="outline"
                    disabled={!statusData.ready}
                    onClick={() => openToolInNewTab("/summary")}
                  >
                    <FileTextIcon className="mr-2 h-4 w-4" />
                    Summary
                    <ExternalLink className="ml-2 h-3 w-3" />
                  </Button>

                  <Button
                    variant="outline"
                    disabled={!statusData.ready}
                    onClick={() => openToolInNewTab("/quiz")}
                  >
                    <HelpCircle className="mr-2 h-4 w-4" />
                    Quiz
                    <ExternalLink className="ml-2 h-3 w-3" />
                  </Button>

                  <Button
                    disabled={!statusData.ready}
                    onClick={() => openToolInNewTab("/flashcards")}
                  >
                    <Brain className="mr-2 h-4 w-4" />
                    Flashcards
                    <ExternalLink className="ml-2 h-3 w-3" />
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {statusData?.ready && (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-primary/10 p-3">
                  <MessageSquare className="h-6 w-6 text-primary" />
                </div>

                <div>
                  <CardTitle className="text-2xl">Ask the Coursework</CardTitle>
                  <CardDescription>
                    Query the indexed material using <code>POST /api/ask</code>.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <Textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Example: summarize the uploaded coursework"
                className="min-h-32"
              />

              <div className="flex flex-wrap gap-3">
                <Button
                  onClick={askRag}
                  disabled={isAsking || !question.trim()}
                >
                  {isAsking ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Asking...
                    </>
                  ) : (
                    <>
                      <Send className="mr-2 h-4 w-4" />
                      Ask RAG
                    </>
                  )}
                </Button>

                <Button
                  variant="outline"
                  onClick={() => setQuestion("summarize the uploaded coursework")}
                  disabled={isAsking}
                >
                  Use summary prompt
                </Button>

                <Button
                  variant="outline"
                  onClick={() => {
                    setQuestion("")
                    setAnswer("")
                    setAskError("")
                  }}
                  disabled={isAsking}
                >
                  Clear
                </Button>
              </div>

              {askError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Ask request failed</AlertTitle>
                  <AlertDescription className="whitespace-pre-wrap">
                    {askError}
                  </AlertDescription>
                </Alert>
              )}

              {answer && (
                <Card className="bg-background">
                  <CardHeader>
                    <CardTitle className="text-base">Answer</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="whitespace-pre-wrap rounded-lg border bg-muted/40 p-4 text-sm leading-6">
                      {answer}
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}