import { Suspense } from "react"
import StatusPageClient from "./status-page-client"

export default function StatusPage() {
  return (
    <Suspense fallback={<main className="p-8">Loading status page...</main>}>
      <StatusPageClient />
    </Suspense>
  )
}