# 校园智搜 — 学校官网 RAG 智能检索系统
这是wiki分支
基于 RAG（检索增强生成）技术的校园知识问答系统。用户输入自然语言问题，系统从学校官网爬取的内容中检索相关信息，返回相关文档片段及 AI 生成的摘要回答，体验类似搜索引擎。

## 系统架构

```
用户查询 → Vue3前端 → FastAPI后端 → RAGChain
                                     ├── Retriever (BGE嵌入 + Chroma向量检索)
                                     └── Generator (DeepSeek LLM + Prompt模板)
                                              ↑
数据管线: Scrapy爬虫 → HTML/PDF解析 → 文本分块 → BGE嵌入 → Chroma+SQLite
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.11+
pip install -r requirements.txt

# Node.js 18+ (前端构建)
cd frontend && npm install
```

### 2. 配置 API Key

编辑项目根目录 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

获取地址：https://platform.deepseek.com

### 3. 数据准备

```bash
# 爬取官网数据
python scripts/crawl.py

# 构建索引
python scripts/index_all.py
```

### 4. 构建前端

```bash
cd frontend
npm run build
```

### 5. 启动服务


日常使用（一条命令）：
cd d:\code\aa-my-idea\school-rag
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
然后浏览器访问 http://127.0.0.1:8000 （推荐用 127.0.0.1 而非 localhost，避免代理干扰）


```bash
# 生产模式（前后端一体）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 开发模式（前后端分离）
# 终端1: 后端
python -m uvicorn api.main:app --reload --port 8000
# 终端2: 前端
cd frontend && npm run dev
```

访问 http://localhost:8000

### 6. 运行集成测试

```bash
# 完整测试（含LLM，约2分钟）
python scripts/integration_test.py

# 快速测试（跳过LLM和浏览器）
python scripts/integration_test.py --skip-llm --skip-browser
```

## 项目结构

```
school-rag/
├── config/          # 全局配置 + 站点爬取规则
├── crawler/         # Scrapy + Playwright 爬虫
├── parser/          # HTML/PDF/OCR 解析器
├── indexer/         # 文本分块 + BGE嵌入 + Chroma/SQLite 存储
├── rag/             # 检索器 + LLM生成器 + RAG Chain
├── api/             # FastAPI 后端接口
├── frontend/        # Vue3 + Vite + TypeScript 搜索界面
├── data/            # 运行时数据（爬取文件、向量库、元数据库）
└── scripts/         # 运维脚本（爬取、索引、测试）
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/search` | 搜索（检索 + LLM生成） |
| POST | `/api/search/stream` | 流式搜索（SSE） |
| GET  | `/api/suggest` | 搜索建议（纯检索） |
| GET  | `/api/stats` | 索引统计 |
| GET  | `/api/health` | 健康检查 |

### 搜索请求示例

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "选课流程是什么？", "top_k": 8, "use_llm": true}'
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI |
| RAG框架 | LangChain |
| LLM | DeepSeek API |
| Embedding | BGE-large-zh-v1.5（本地运行） |
| 向量数据库 | Chroma |
| 元数据库 | SQLite |
| 爬虫 | Scrapy + Playwright |
| 文档解析 | BeautifulSoup4 + PyMuPDF + pdfplumber |
| 前端 | Vue3 + Vite + TypeScript |
| HTTP客户端 | Axios |

## 集成测试结果

```
30 项测试全部通过：
  配置与环境: 3/3 ✅
  数据层:     4/4 ✅ (257 HTML, 45 PDF, 1240文档, 1928分块)
  解析层:     3/3 ✅ (HTML/PDF解析器, 路由器)
  索引层:     4/4 ✅ (BGE 1024维, Chroma 1928块, 检索正常)
  RAG层:      4/4 ✅ (纯检索+LLM生成, 来源溯源)
  API层:      8/8 ✅ (健康/统计/搜索/建议/前端/CORS)
  浏览器E2E:  4/4 ✅ (页面加载/搜索框/搜索/fetch)
```
