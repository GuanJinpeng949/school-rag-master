<template>
  <div class="app">
    <header class="header">
      <div class="header-inner">
        <div class="logo" @click="resetSearch">
          <span class="logo-icon">🎓</span>
          <span>校园智能问答</span>
        </div>
        <div class="header-stats" v-if="stats">
          已索引 {{ stats.total_documents }} 篇文档 · {{ stats.chroma_chunks }} 个分块
        </div>
      </div>
    </header>

    <!-- 搜索首页 -->
    <div v-if="!hasSearched" class="hero">
      <h1>校园知识，一问即答</h1>
      <p>基于学校官网内容的智能检索问答系统，支持来源溯源</p>
      <SearchBox
        v-model="query"
        :loading="loading"
        @search="doSearch"
      />
      <div class="suggestions">
        <span
          v-for="s in suggestions"
          :key="s"
          class="suggest-tag"
          @click="query = s; doSearch()"
        >{{ s }}</span>
      </div>
    </div>

    <!-- 搜索结果页 -->
    <div v-else class="main-content">
      <SearchBox
        v-model="query"
        :loading="loading"
        @search="doSearch"
      />

      <!-- 加载中（检索阶段） -->
      <div v-if="loading && !streamAnswer" style="text-align: center; padding: 40px 0;">
        <div class="loading-spinner"></div>
        <p style="margin-top: 12px; color: var(--text-secondary);">
          正在检索相关文档...
        </p>
      </div>

      <!-- 错误 -->
      <div v-else-if="error" class="answer-card" style="border-left-color: var(--error);">
        <p style="color: var(--error);">{{ error }}</p>
      </div>

      <!-- 结果 -->
      <template v-else-if="response || streamAnswer">
        <!-- AI回答（流式） -->
        <div v-if="streamAnswer" class="answer-card">
          <h3>
            <span v-if="hasLlm">🤖 AI 回答</span>
            <span v-else>📋 检索结果</span>
            <span v-if="isStreaming" class="streaming-badge">生成中...</span>
          </h3>
          <div class="answer-content" v-html="renderedAnswer"></div>
          <div v-if="isStreaming" class="cursor-blink">▊</div>
        </div>

        <!-- 来源链接 -->
        <div v-if="streamSources && streamSources.length" class="sources-section">
          <h4>📎 相关链接</h4>
          <div v-for="(s, i) in streamSources" :key="i" class="source-link">
            <span class="site-badge">{{ s.site || '来源' }}</span>
            <a :href="s.url" target="_blank" rel="noopener">{{ s.title || s.url }}</a>
          </div>
        </div>

        <!-- 状态栏 -->
        <div class="status-bar">
          <div class="time-info">
            <span v-if="retrieveTimeMs">检索: {{ retrieveTimeMs }}ms</span>
            <span v-if="generateTimeMs">生成: {{ generateTimeMs }}ms</span>
            <span v-if="totalTimeMs">总耗时: {{ totalTimeMs }}ms</span>
          </div>
          <span :class="['status-badge', hasLlm ? 'llm' : 'fallback']">
            {{ hasLlm ? 'AI摘要' : '纯检索' }}
          </span>
        </div>

        <!-- 详细结果 -->
        <details v-if="streamResults.length" style="margin-top: 16px;">
          <summary style="cursor: pointer; color: var(--text-secondary); font-size: 14px;">
            查看检索详情 ({{ streamResults.length }} 条)
          </summary>
          <ResultItem
            v-for="(r, i) in streamResults"
            :key="i"
            :result="r"
          />
        </details>
      </template>
    </div>

    <footer class="footer">
      校园智能问答系统 · 基于RAG技术 · 来源可溯源
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import MarkdownIt from 'markdown-it'
import SearchBox from './components/SearchBox.vue'
import ResultItem from './components/ResultItem.vue'
import { searchStream, getStats } from './api'
import type { StatsResponse, SourceItem, ResultItem as ResultItemType } from './api/types'

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
})

const query = ref('')
const loading = ref(false)
const hasSearched = ref(false)
const error = ref('')
const stats = ref<StatsResponse | null>(null)

// 流式状态
const streamAnswer = ref('')
const streamSources = ref<SourceItem[]>([])
const streamResults = ref<ResultItemType[]>([])
const isStreaming = ref(false)
const hasLlm = ref(false)
const retrieveTimeMs = ref(0)
const generateTimeMs = ref(0)
const totalTimeMs = ref(0)

// 保留 response 用于兼容
const response = ref<any>(null)

// 流式请求控制器
let abortController: AbortController | null = null

const suggestions = [
  '选课流程',
  '奖学金申请条件',
  '保研政策',
  '图书馆开放时间',
  '转专业流程',
  '毕业学分要求',
]

const renderedAnswer = computed(() => {
  if (!streamAnswer.value) return ''
  return md.render(streamAnswer.value)
})

function resetSearch() {
  hasSearched.value = false
  response.value = null
  streamAnswer.value = ''
  streamSources.value = []
  streamResults.value = []
  error.value = ''
  query.value = ''
  isStreaming.value = false
  retrieveTimeMs.value = 0
  generateTimeMs.value = 0
  totalTimeMs.value = 0
}

async function doSearch() {
  const q = query.value.trim()
  if (!q || loading.value) return

  // 取消上一个流式请求
  if (abortController) {
    abortController.abort()
    abortController = null
  }

  loading.value = true
  error.value = ''
  hasSearched.value = true
  response.value = null
  streamAnswer.value = ''
  streamSources.value = []
  streamResults.value = []
  isStreaming.value = false
  hasLlm.value = true
  retrieveTimeMs.value = 0
  generateTimeMs.value = 0
  totalTimeMs.value = 0

  const startTime = Date.now()

  try {
    abortController = searchStream(
      {
        query: q,
        top_k: 5,
        use_llm: true,
      },
      // onMeta: 检索完成，立即展示来源
      (meta) => {
        console.log('[App] Received meta:', meta)
        retrieveTimeMs.value = meta.retrieve_time_ms || 0
        streamSources.value = meta.sources || []
        streamResults.value = meta.results || []
        hasLlm.value = meta.has_llm !== false
        isStreaming.value = true
      },
      // onToken: 流式接收生成内容
      (token) => {
        streamAnswer.value += token
      },
      // onDone: 生成完成
      (genMs) => {
        generateTimeMs.value = genMs
        totalTimeMs.value = Date.now() - startTime
        isStreaming.value = false
        loading.value = false
      },
      // onError
      (errMsg) => {
        console.error('[App] Stream error:', errMsg)
        error.value = errMsg
        isStreaming.value = false
        loading.value = false
      },
    )
  } catch (e: any) {
    console.error('[App] Search error:', e)
    const msg = e?.message || String(e)
    error.value = `请求失败: ${msg}`
    loading.value = false
    isStreaming.value = false
  }
}

onMounted(async () => {
  try {
    stats.value = await getStats()
  } catch {
    // ignore
  }
})

onUnmounted(() => {
  if (abortController) {
    abortController.abort()
  }
})
</script>

<style scoped>
.streaming-badge {
  font-size: 12px;
  font-weight: 400;
  color: var(--primary);
  margin-left: 8px;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.cursor-blink {
  display: inline;
  color: var(--primary);
  animation: blink-cursor 0.8s step-end infinite;
  font-weight: bold;
}

@keyframes blink-cursor {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
