export interface SearchRequest {
  query: string
  top_k?: number
  filter_site?: string
  filter_type?: string
  use_llm?: boolean
}

export interface SourceItem {
  url: string
  title: string
  site: string
  type: string
}

export interface ResultItem {
  text: string
  source_url: string
  source_site: string
  title: string
  content_type: string
  publish_date: string
  score: number
  doc_id: string
  chunk_index: number
  total_chunks: number
}

export interface SearchResponse {
  query: string
  answer: string
  results: ResultItem[]
  sources: SourceItem[]
  has_llm: boolean
  error: string | null
  retrieve_time_ms: number
  generate_time_ms: number
  total_time_ms: number
}

export interface StatsResponse {
  total_documents: number
  total_chunks: number
  total_text_length: number
  indexed_documents: number
  by_site: Record<string, number>
  by_type: Record<string, number>
  chroma_chunks: number
  site_distribution: Record<string, number>
}

export interface HealthResponse {
  status: string
  version: string
  chroma_ok: boolean
  sqlite_ok: boolean
}
