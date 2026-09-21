<template>
  <div class="search-box">
    <span class="search-icon">🔍</span>
    <input
      ref="inputRef"
      v-model="modelValue"
      class="search-input"
      type="text"
      placeholder="输入你的问题，如：选课流程是什么？"
      @keydown.enter="$emit('search')"
    />
    <button
      class="search-btn"
      :disabled="loading || !modelValue.trim()"
      @click="$emit('search')"
    >
      {{ loading ? '搜索中...' : '搜索' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

defineProps<{
  loading: boolean
}>()

const modelValue = defineModel<string>({ required: true })
const inputRef = ref<HTMLInputElement | null>(null)

defineEmits<{
  search: []
}>()

onMounted(() => {
  inputRef.value?.focus()
})
</script>
