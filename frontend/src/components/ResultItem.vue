<template>
  <div class="result-item">
    <div class="result-title">
      <a :href="result.source_url" target="_blank" rel="noopener">
        {{ result.title || '无标题' }}
      </a>
    </div>
    <div class="result-meta">
      <span class="source-tag">{{ result.source_site || '未知来源' }}</span>
      <span v-if="result.publish_date">{{ result.publish_date }}</span>
      <span class="result-score">相关度 {{ (result.score * 100).toFixed(1) }}%</span>
    </div>
    <div class="result-text">{{ result.text }}</div>
  </div>
</template>

<script setup lang="ts">
import type { ResultItem } from '../api/types'

defineProps<{
  result: ResultItem
}>()
</script>
