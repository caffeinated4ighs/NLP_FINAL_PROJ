import { Suspense } from "react"
import FlashcardsPageClient from "./flashcards-page-client"

export default function FlashcardsPage() {
  return (
    <Suspense fallback={<main className="p-8">Loading flashcards...</main>}>
      <FlashcardsPageClient />
    </Suspense>
  )
}