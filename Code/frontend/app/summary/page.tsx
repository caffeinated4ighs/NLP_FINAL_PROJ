import { Suspense } from "react"
import SummaryPageClient from "./summary-page-client"

export default function SummaryPage() {
  return (
    <Suspense fallback={<main className="p-8">Loading summary...</main>}>
      <SummaryPageClient />
    </Suspense>
  )
}