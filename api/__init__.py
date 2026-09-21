"""API模块 - FastAPI后端服务

路由:
- POST /api/search   - 搜索（检索+LLM生成）
- POST /api/search/stream - 流式搜索
- GET  /api/suggest  - 搜索建议（纯检索）
- GET  /api/stats    - 索引统计
- GET  /api/health   - 健康检查
"""
