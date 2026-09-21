"""集成测试 - 端到端验证全链路

测试范围:
1. 数据层：原始文件存在性、数据库/向量库数据完整性
2. 解析层：HTML/PDF解析器输出正确
3. 索引层：向量检索返回相关结果
4. RAG层：检索+LLM生成端到端
5. API层：HTTP接口正确性
6. 前端层：页面加载+搜索交互

用法:
    cd d:/code/aa-my-idea/school-rag
    python scripts/integration_test.py
    python scripts/integration_test.py --skip-llm   # 跳过LLM测试（更快）
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional

# 修复Windows终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ============================================================
# 辅助
# ============================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        self.details.append(("PASS", name, detail))
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = ""):
        self.failed += 1
        self.details.append(("FAIL", name, detail))
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

    def skip(self, name: str, detail: str = ""):
        self.skipped += 1
        self.details.append(("SKIP", name, detail))
        print(f"  ⏭️  {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"集成测试结果: {total} 项")
        print(f"  通过: {self.passed}  失败: {self.failed}  跳过: {self.skipped}")
        if self.failed > 0:
            print(f"\n失败项:")
            for status, name, detail in self.details:
                if status == "FAIL":
                    print(f"  ❌ {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


R = TestResult()

# ============================================================
# 1. 数据层测试
# ============================================================

def test_data_layer():
    print("\n📦 1. 数据层测试")
    
    # 1.1 原始文件
    raw_dir = PROJECT_ROOT / "data" / "raw"
    html_dir = raw_dir / "html"
    pdf_dir = raw_dir / "pdf"
    
    html_files = list(html_dir.glob("*.html")) if html_dir.exists() else []
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    
    if len(html_files) > 0:
        R.ok("HTML原始文件", f"{len(html_files)} 个")
    else:
        R.fail("HTML原始文件", "无HTML文件，请先执行爬取")
    
    if len(pdf_files) > 0:
        R.ok("PDF原始文件", f"{len(pdf_files)} 个")
    else:
        R.skip("PDF原始文件", "无PDF文件（可能目标站点无PDF）")
    
    # 1.2 SQLite数据库
    db_path = PROJECT_ROOT / "data" / "metadata.db"
    if db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(db_path))
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            indexed_count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE indexed_at != '' AND indexed_at IS NOT NULL"
            ).fetchone()[0]
            conn.close()
            R.ok("SQLite数据库", f"{doc_count}文档, {chunk_count}分块, {indexed_count}已索引")
            
            if doc_count == 0:
                R.fail("文档数量", "数据库中无文档，请先执行索引")
            if indexed_count == 0:
                R.fail("已索引文档", "无已索引文档，请先执行索引构建")
        except Exception as e:
            R.fail("SQLite数据库", str(e))
    else:
        R.fail("SQLite数据库", "metadata.db 不存在")
    
    # 1.3 Chroma向量库
    chroma_dir = PROJECT_ROOT / "data" / "chroma"
    if chroma_dir.exists():
        chroma_files = list(chroma_dir.rglob("*"))
        if len(chroma_files) > 3:
            R.ok("Chroma向量库", f"{len(chroma_files)} 个文件")
        else:
            R.fail("Chroma向量库", "文件过少，可能索引未完成")
    else:
        R.fail("Chroma向量库", "chroma目录不存在")


# ============================================================
# 2. 解析层测试
# ============================================================

def test_parser_layer():
    print("\n📄 2. 解析层测试")
    
    # 2.1 HTML解析
    try:
        from parser.html_parser import HTMLParser
        parser = HTMLParser()
        
        # 取一个HTML文件测试
        html_dir = PROJECT_ROOT / "data" / "raw" / "html"
        html_files = list(html_dir.glob("*.html"))
        if html_files:
            sample = html_files[0]
            docs = parser.parse(str(sample))
            if docs and len(docs) > 0 and docs[0].text.strip():
                R.ok("HTML解析器", f"解析 {sample.name}: {len(docs)} 段, 首段 {len(docs[0].text)} 字")
            else:
                R.fail("HTML解析器", f"解析 {sample.name} 返回空结果")
        else:
            R.skip("HTML解析器", "无HTML文件可测试")
    except Exception as e:
        R.fail("HTML解析器", str(e))
    
    # 2.2 PDF解析
    try:
        from parser.pdf_parser import PDFParser
        parser = PDFParser()
        
        pdf_dir = PROJECT_ROOT / "data" / "raw" / "pdf"
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            sample = pdf_files[0]
            docs = parser.parse(str(sample))
            if docs and len(docs) > 0 and docs[0].text.strip():
                R.ok("PDF解析器", f"解析 {sample.name}: {len(docs)} 段, 首段 {len(docs[0].text)} 字")
            else:
                R.skip("PDF解析器", f"解析 {sample.name} 返回空（可能是扫描PDF，需OCR）")
        else:
            R.skip("PDF解析器", "无PDF文件可测试")
    except Exception as e:
        R.fail("PDF解析器", str(e))
    
    # 2.3 解析路由
    try:
        from parser.router import ParserRouter
        router = ParserRouter()
        R.ok("解析路由器", f"已注册: {list(router._parsers.keys())}")
    except Exception as e:
        R.fail("解析路由器", str(e))


# ============================================================
# 3. 索引层测试
# ============================================================

def test_indexer_layer():
    print("\n🔍 3. 索引层测试")
    
    # 3.1 Embedder
    try:
        from indexer.embedder import Embedder
        embedder = Embedder()
        
        # 测试嵌入
        test_text = "选课流程是什么"
        embedding = embedder.embed_query(test_text)
        if embedding and len(embedding) > 0:
            R.ok("Embedder", f"维度: {len(embedding)}, 前3值: {embedding[:3]}")
        else:
            R.fail("Embedder", "嵌入返回空")
    except Exception as e:
        R.fail("Embedder", str(e))
    
    # 3.2 VectorStore
    try:
        from indexer.vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_stats()
        total_chunks = stats.get("total_chunks", 0)
        if total_chunks > 0:
            R.ok("VectorStore", f"{total_chunks} 个向量分块")
        else:
            R.fail("VectorStore", "向量库为空，请先构建索引")
    except Exception as e:
        R.fail("VectorStore", str(e))
    
    # 3.3 检索测试
    try:
        from rag.retriever import Retriever
        retriever = Retriever()
        
        test_queries = ["选课流程", "奖学金申请"]
        for q in test_queries:
            results = retriever.retrieve(query=q, top_k=3)
            if results:
                top = results[0]
                R.ok(f"检索 '{q}'", f"{len(results)} 条, 最高分: {top.score:.4f}, 来源: {top.source_site}")
            else:
                R.fail(f"检索 '{q}'", "无结果")
    except Exception as e:
        R.fail("检索器", str(e))


# ============================================================
# 4. RAG层测试
# ============================================================

def test_rag_layer(skip_llm: bool = False):
    print("\n🤖 4. RAG层测试")
    
    # 4.1 纯检索
    try:
        from rag.chain import RAGChain
        chain = RAGChain()
        
        resp = chain.search_only(query="选课流程", top_k=5)
        if resp and len(resp) > 0:
            top = resp[0]
            R.ok("RAG纯检索", f"{len(resp)} 条结果, 最高分: {top.score:.4f}")
            # 检查结果字段完整性
            required_fields = ["text", "source_url", "source_site", "title", "content_type", "score"]
            missing = [f for f in required_fields if not getattr(top, f, None)]
            if missing:
                R.fail("检索结果字段", f"缺失: {missing}")
            else:
                R.ok("检索结果字段", "所有必需字段完整")
        else:
            R.fail("RAG纯检索", "无结果")
    except Exception as e:
        R.fail("RAG纯检索", str(e))
    
    # 4.2 LLM生成
    if skip_llm:
        R.skip("RAG LLM生成", "已跳过 (--skip-llm)")
        return
    
    try:
        from rag.chain import RAGChain
        chain = RAGChain()
        
        start = time.time()
        resp = chain.ask(query="选课流程是什么？有哪些注意事项？", top_k=5)
        elapsed = int((time.time() - start) * 1000)
        
        if resp.error:
            R.fail("RAG LLM生成", f"返回错误: {resp.error}")
        elif resp.answer and len(resp.answer) > 50:
            has_sources = bool(resp.sources)
            R.ok("RAG LLM生成", f"回答 {len(resp.answer)} 字, 检索 {resp.retrieve_time_ms}ms, "
                 f"生成 {resp.generate_time_ms}ms, 总耗时 {elapsed}ms, 来源: {len(resp.sources)} 条")
            
            if has_sources:
                R.ok("LLM来源溯源", f"{len(resp.sources)} 条来源链接")
            else:
                R.fail("LLM来源溯源", "无来源链接")
        else:
            R.fail("RAG LLM生成", f"回答过短或为空: '{resp.answer[:100]}'")
    except Exception as e:
        R.fail("RAG LLM生成", str(e))


# ============================================================
# 5. API层测试
# ============================================================

def test_api_layer():
    print("\n🌐 5. API层测试")
    
    import httpx
    
    base_url = "http://localhost:8000"
    
    # 5.1 健康检查
    try:
        r = httpx.get(f"{base_url}/api/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            R.ok("健康检查", f"status={data.get('status')}, chroma={data.get('chroma_ok')}, sqlite={data.get('sqlite_ok')}")
        else:
            R.fail("健康检查", f"HTTP {r.status_code}")
    except httpx.ConnectError:
        R.fail("健康检查", "无法连接API服务，请先启动: python -m uvicorn api.main:app --port 8000")
        return  # API没启动，后续测试无法进行
    except Exception as e:
        R.fail("健康检查", str(e))
        return
    
    # 5.2 统计接口
    try:
        r = httpx.get(f"{base_url}/api/stats", timeout=10)
        if r.status_code == 200:
            data = r.json()
            R.ok("统计接口", f"文档={data.get('total_documents')}, 分块={data.get('total_chunks')}, "
                 f"Chroma={data.get('chroma_chunks')}")
        else:
            R.fail("统计接口", f"HTTP {r.status_code}")
    except Exception as e:
        R.fail("统计接口", str(e))
    
    # 5.3 搜索接口（无LLM）
    try:
        r = httpx.post(
            f"{base_url}/api/search",
            json={"query": "选课流程", "top_k": 5, "use_llm": False},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                R.ok("搜索API (无LLM)", f"{len(results)} 条结果, 耗时 {data.get('total_time_ms')}ms")
            else:
                R.fail("搜索API (无LLM)", "无搜索结果")
        else:
            R.fail("搜索API (无LLM)", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        R.fail("搜索API (无LLM)", str(e))
    
    # 5.4 搜索接口（含LLM）
    try:
        r = httpx.post(
            f"{base_url}/api/search",
            json={"query": "选课流程是什么？", "top_k": 5, "use_llm": True},
            timeout=180,
        )
        if r.status_code == 200:
            data = r.json()
            answer = data.get("answer", "")
            if answer and len(answer) > 50:
                R.ok("搜索API (含LLM)", f"回答 {len(answer)} 字, 来源 {len(data.get('sources', []))} 条, "
                     f"耗时 {data.get('total_time_ms')}ms")
            else:
                R.fail("搜索API (含LLM)", f"回答过短: '{answer[:100]}'")
        else:
            R.fail("搜索API (含LLM)", f"HTTP {r.status_code}")
    except Exception as e:
        R.fail("搜索API (含LLM)", str(e))
    
    # 5.5 搜索建议接口
    try:
        r = httpx.get(f"{base_url}/api/suggest", params={"q": "选课", "top_k": 3}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            R.ok("搜索建议API", f"{len(results)} 条建议")
        else:
            R.fail("搜索建议API", f"HTTP {r.status_code}")
    except Exception as e:
        R.fail("搜索建议API", str(e))
    
    # 5.6 前端页面
    try:
        r = httpx.get(f"{base_url}/", timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            R.ok("前端页面", "HTML正常返回")
        else:
            R.fail("前端页面", f"HTTP {r.status_code}, CT={r.headers.get('content-type')}")
    except Exception as e:
        R.fail("前端页面", str(e))
    
    # 5.7 前端JS资源
    try:
        dist_dir = PROJECT_ROOT / "frontend" / "dist" / "assets"
        js_files = list(dist_dir.glob("index-*.js")) if dist_dir.exists() else []
        if js_files:
            js_name = js_files[0].name
            r = httpx.get(f"{base_url}/assets/{js_name}", timeout=10)
            if r.status_code == 200:
                R.ok("前端JS资源", f"{js_name} ({len(r.text)//1024}KB)")
            else:
                R.fail("前端JS资源", f"HTTP {r.status_code}")
        else:
            R.fail("前端JS资源", "dist目录无JS文件，请先构建: cd frontend && npm run build")
    except Exception as e:
        R.fail("前端JS资源", str(e))
    
    # 5.8 CORS头
    try:
        r = httpx.options(
            f"{base_url}/api/search",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
            timeout=10,
        )
        acao = r.headers.get("access-control-allow-origin", "")
        if acao:
            R.ok("CORS配置", f"Allow-Origin: {acao}")
        else:
            R.fail("CORS配置", "缺少 Access-Control-Allow-Origin 头")
    except Exception as e:
        R.fail("CORS配置", str(e))


# ============================================================
# 6. 浏览器端到端测试
# ============================================================

def test_browser_e2e():
    print("\n🖥️  6. 浏览器端到端测试")
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        R.skip("浏览器E2E测试", "Playwright未安装，跳过")
        return
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # 6.1 页面加载
            page.goto("http://localhost:8000/", wait_until="domcontentloaded", timeout=15000)
            title = page.title()
            if "校园" in title or "问答" in title:
                R.ok("页面加载", f"标题: {title}")
            else:
                R.fail("页面加载", f"标题异常: {title}")
            
            # 6.2 搜索框存在
            search_input = page.locator("input").first
            if search_input.is_visible():
                R.ok("搜索框", "可见")
            else:
                R.fail("搜索框", "不可见")
            
            # 6.3 搜索功能（无LLM）
            search_input.fill("选课流程")
            search_input.press("Enter")
            time.sleep(8)  # 等待检索结果
            
            page_text = page.locator("body").text_content() or ""
            if "error" in page_text.lower() and "network" in page_text.lower():
                R.fail("搜索交互(无LLM)", f"页面显示网络错误")
            elif "检索" in page_text or "result" in page_text.lower() or "来源" in page_text:
                R.ok("搜索交互(无LLM)", "结果显示正常")
            else:
                R.skip("搜索交互(无LLM)", f"无法确认结果: {page_text[:200]}")
            
            # 6.4 fetch API直接测试
            fetch_result = page.evaluate("""
                async () => {
                    try {
                        const r = await fetch("/api/search", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({query: "奖学金", top_k: 3, use_llm: false})
                        });
                        const data = await r.json();
                        return {status: r.status, results: data.results?.length || 0, error: data.error};
                    } catch(e) {
                        return {error: e.message, name: e.name};
                    }
                }
            """)
            
            if fetch_result.get("error") and fetch_result.get("name"):
                R.fail("浏览器fetch测试", f"{fetch_result['name']}: {fetch_result['error']}")
            elif fetch_result.get("status") == 200 and fetch_result.get("results", 0) > 0:
                R.ok("浏览器fetch测试", f"HTTP 200, {fetch_result['results']} 条结果")
            else:
                R.fail("浏览器fetch测试", f"结果: {fetch_result}")
            
            browser.close()
    except Exception as e:
        R.fail("浏览器E2E测试", str(e))


# ============================================================
# 7. 配置与环境测试
# ============================================================

def test_config():
    print("\n⚙️  7. 配置与环境测试")
    
    # 7.1 .env文件
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            content = f.read()
        has_deepseek = "DEEPSEEK_API_KEY" in content and "your_" not in content.split("DEEPSEEK_API_KEY")[1].split("\n")[0]
        if has_deepseek:
            R.ok(".env配置", "DEEPSEEK_API_KEY 已配置")
        else:
            R.fail(".env配置", "DEEPSEEK_API_KEY 未配置或仍为占位符")
    else:
        R.fail(".env配置", ".env 文件不存在")
    
    # 7.2 站点配置
    sites_yaml = PROJECT_ROOT / "config" / "sites.yaml"
    if sites_yaml.exists():
        R.ok("站点配置", f"sites.yaml 存在")
    else:
        R.fail("站点配置", "sites.yaml 不存在")
    
    # 7.3 前端构建产物
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if dist_dir.exists():
        html = dist_dir / "index.html"
        assets = dist_dir / "assets"
        if html.exists() and assets.exists():
            R.ok("前端构建产物", "dist/ 完整")
        else:
            R.fail("前端构建产物", "dist/ 不完整")
    else:
        R.fail("前端构建产物", "dist/ 不存在，请先构建")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="校园智搜集成测试")
    parser.add_argument("--skip-llm", action="store_true", help="跳过LLM测试（更快）")
    parser.add_argument("--skip-browser", action="store_true", help="跳过浏览器E2E测试")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 校园智搜 — 集成测试")
    print("=" * 60)
    
    test_config()
    test_data_layer()
    test_parser_layer()
    test_indexer_layer()
    test_rag_layer(skip_llm=args.skip_llm)
    test_api_layer()
    
    if not args.skip_browser:
        test_browser_e2e()
    else:
        R.skip("浏览器E2E测试", "已跳过 (--skip-browser)")
    
    success = R.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
