import axios from 'axios'
import type { SearchRequest, SearchResponse, StatsResponse, HealthResponse } from './types'

const api = axios.create({
  baseURL: '/api',
  timeout: 180000, // 3分钟，LLM生成可能较慢
})

// 响应拦截器：记录详细错误
api.interceptors.response.use(
  response => {
    console.log('[API] Response:', response.config.url, response.status)
    return response
  },
  error => {
    console.error('[API] Error:', error.message)
    console.error('[API] Config:', error.config?.url, error.config?.method)
    console.error('[API] Response:', error.response?.status, error.response?.data)
    return Promise.reject(error)
  }
)

export async function search(req: SearchRequest): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>('/search', req)
  return data
}

/** 流式搜索 - 返回一个可取消的流式控制器 */
export function searchStream(
  req: SearchRequest,
  onMeta: (meta: { retrieve_time_ms: number; sources: any[]; results: any[]; has_llm: boolean }) => void,
  onToken: (token: string) => void,
  onDone: (generate_time_ms: number) => void,
  onError: (error: string) => void,
): AbortController {
  const controller = new AbortController()

  const doStream = async () => {
    try {
      const response = await fetch('/api/search/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal: controller.signal,
      })

      if (!response.body) {
        onError('浏览器不支持流式响应')
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue

          try {
            const event = JSON.parse(raw)

            if (event.type === 'meta') {
              onMeta(event)
            } else if (event.type === 'token') {
              onToken(event.content)
            } else if (event.type === 'done') {
              onDone(event.generate_time_ms || 0)
            } else if (event.type === 'error') {
              onError(event.message || '未知错误')
            }
          } catch {
            // 兼容旧格式: data: {"content":"..."}
            try {
              const legacy = JSON.parse(raw)
              if (legacy.content) onToken(legacy.content)
              if (legacy.error) onError(legacy.error)
            } catch {
              // skip
            }
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        onError(e.message || '流式请求失败')
      }
    }
  }

  doStream()
  return controller
}

export async function suggest(query: string, topK = 5): Promise<StatsResponse> {
  const { data } = await api.get<StatsResponse>('/suggest', {
    params: { q: query, top_k: topK },
  })
  return data
}

export async function getStats(): Promise<StatsResponse> {
  const { data } = await api.get<StatsResponse>('/stats')
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}
