import { Suspense } from "react"
import QuizPageClient from "./quiz-page-client"

export default function QuizPage() {
  return (
    <Suspense fallback={<main className="p-8">Loading quiz...</main>}>
      <QuizPageClient />
    </Suspense>
  )
}