"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Upload, FileText, Loader2, CheckCircle2, AlertCircle } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"

type UploadResponse = {
  session_id: string
  uploaded_files: string[]
}

type InitResponse = {
  session_id: string
  status: string
  ready?: boolean
  message?: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

export default function UploadPage() {
  const router = useRouter()

  const [files, setFiles] = useState<File[]>([])
  const [sessionId, setSessionId] = useState("")
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [isWorking, setIsWorking] = useState(false)
  const [phase, setPhase] = useState<"idle" | "uploading" | "initializing" | "done">("idle")
  const [error, setError] = useState("")

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files || [])
    setFiles(selectedFiles)
    setError("")
    setSessionId("")
    setUploadedFiles([])
    setPhase("idle")
  }

  async function uploadFiles(): Promise<UploadResponse> {
    const formData = new FormData()

    for (const file of files) {
      formData.append("files", file)
    }

    const response = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || "Upload failed.")
    }

    return response.json()
  }

  async function initializeSession(sessionId: string): Promise<InitResponse> {
    const response = await fetch(`${API_BASE}/api/init`, {
      method: "POST",
      headers: {
        accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        use_ocr_for_all_pdfs: false,
        include_video_frame_ocr: true,
        video_frame_interval_sec: 10,
      }),
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || "Initialization failed.")
    }

    return response.json()
  }

  async function handleUploadAndInit() {
    if (files.length === 0) {
      setError("Please select at least one file.")
      return
    }

    setIsWorking(true)
    setError("")
    setSessionId("")
    setUploadedFiles([])

    try {
      setPhase("uploading")
      const uploadData = await uploadFiles()

      localStorage.setItem("rag_session_id", uploadData.session_id)
      localStorage.setItem("rag_uploaded_files", JSON.stringify(uploadData.uploaded_files))

      setSessionId(uploadData.session_id)
      setUploadedFiles(uploadData.uploaded_files)

      setPhase("initializing")
      await initializeSession(uploadData.session_id)

      setPhase("done")

      router.push(`/status?session_id=${uploadData.session_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error.")
      setPhase("idle")
    } finally {
      setIsWorking(false)
    }
  }

  return (
    <main className="min-h-screen bg-muted/40 px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <Card className="shadow-sm">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-primary/10 p-3">
                <Upload className="h-6 w-6 text-primary" />
              </div>

              <div>
                <CardTitle className="text-2xl">Upload Coursework</CardTitle>
                <CardDescription>
                  Upload one or more files. After upload, the backend will start indexing.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            <Input
              type="file"
              multiple
              accept=".pdf,.png,.jpg,.jpeg,.mp4,.mov,.txt,.md"
              onChange={handleFileChange}
            />

            {files.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">Selected files</h3>
                  <Badge variant="secondary">{files.length} file(s)</Badge>
                </div>

                <div className="rounded-lg border bg-background">
                  {files.map((file, index) => (
                    <div key={`${file.name}-${index}`}>
                      <div className="flex items-center justify-between gap-4 p-3">
                        <div className="flex min-w-0 items-center gap-3">
                          <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                          <span className="truncate text-sm">{file.name}</span>
                        </div>

                        <span className="shrink-0 text-xs text-muted-foreground">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>

                      {index < files.length - 1 && <Separator />}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <Button
              onClick={handleUploadAndInit}
              disabled={isWorking || files.length === 0}
              className="w-full"
            >
              {isWorking ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {phase === "uploading" && "Uploading files..."}
                  {phase === "initializing" && "Starting backend initialization..."}
                  {phase === "done" && "Redirecting..."}
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload + Initialize
                </>
              )}
            </Button>

            {phase !== "idle" && !error && (
              <Alert>
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Progress</AlertTitle>
                <AlertDescription>
                  {phase === "uploading" && "Uploading files to backend..."}
                  {phase === "initializing" &&
                    "Upload complete. Backend initialization has started."}
                  {phase === "done" && "Redirecting to status page..."}

                  {sessionId && (
                    <div className="mt-2">
                      <span className="font-medium">Session ID:</span>{" "}
                      <code className="rounded bg-muted px-1 py-0.5">{sessionId}</code>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Request failed</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {uploadedFiles.length > 0 && (
              <div className="rounded-lg border bg-background p-4">
                <h3 className="mb-2 font-medium">Uploaded files</h3>
                <ul className="list-inside list-disc space-y-1 text-sm text-muted-foreground">
                  {uploadedFiles.map((filePath) => (
                    <li key={filePath} className="break-all">
                      {filePath}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  )
}