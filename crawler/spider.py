"""爬虫主逻辑 - 基于requests+BeautifulSoup实现轻量级爬取

支持特性:
- 多站点配置(从sites.yaml读取)
- 增量爬取(基于URL去重)
- 多格式文件下载(HTML/PDF/图片/Office文档)
- 公告详情页深度解析(标题/正文/日期/附件)
- PDF附件自动发现和下载
- 爬取结果存储(原始文件+元数据)
"""
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from config.settings import settings, RAW_DATA_DIR


@dataclass
class CrawlResult:
    """爬取结果"""
    url: str
    title: str
    content_type: str  # html, pdf, image, doc, other
    file_path: Optional[str] = None  # 本地存储路径
    file_hash: Optional[str] = None  # 文件哈希
    status_code: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    crawled_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SchoolCrawler:
    """学校官网爬虫"""

    # PDF附件常见Content-Type
    PDF_CONTENT_TYPES = {"application/pdf"}
    # Office文档常见Content-Type
    OFFICE_CONTENT_TYPES = {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    def __init__(self, site_config: dict):
        self.site_config = site_config
        self.name = site_config.get("name", "unknown")
        self.base_url = site_config.get("base_url", "")
        self.allowed_domains = site_config.get("allowed_domains", [])
        self.start_urls = site_config.get("start_urls", [self.base_url])
        self.depth = site_config.get("depth", 2)
        self.delay = site_config.get("delay", 2.0)
        self.js_render = site_config.get("js_render", False)
        self.url_patterns = [re.compile(p) for p in site_config.get("url_patterns", [".*"])]
        self.exclude_patterns = [re.compile(p) for p in site_config.get("exclude_patterns", [])]

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.crawler.crawl_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

        # 已访问URL集合
        self.visited_urls: set[str] = set()
        # 待访问URL队列: (url, current_depth)
        self.queue: list[tuple[str, int]] = []
        # 爬取结果
        self.results: list[CrawlResult] = []

        # 确保存储目录
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保存储目录存在"""
        for subdir in ["html", "pdf", "images", "doc", "other"]:
            (RAW_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

    def _is_allowed_url(self, url: str) -> bool:
        """检查URL是否在允许爬取范围内"""
        parsed = urlparse(url)

        # 检查域名
        domain_allowed = any(domain in parsed.netloc for domain in self.allowed_domains)
        if not domain_allowed:
            return False

        # 检查排除模式
        for pattern in self.exclude_patterns:
            if pattern.search(url):
                return False

        # 检查包含模式
        for pattern in self.url_patterns:
            if pattern.search(url):
                return True

        return False

    def _classify_url(self, url: str) -> str:
        """根据URL判断内容类型"""
        url_lower = url.lower()
        # 提取路径部分（去掉查询参数）
        path = urlparse(url_lower).path.lower()

        if path.endswith(".pdf"):
            return "pdf"
        elif any(path.endswith(ext) for ext in [".doc", ".docx"]):
            return "doc"
        elif any(path.endswith(ext) for ext in [".xls", ".xlsx", ".ppt", ".pptx"]):
            return "doc"
        elif any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            return "image"
        else:
            return "html"

    def _compute_hash(self, content: bytes) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content).hexdigest()[:16]

    def _save_file(self, url: str, content: bytes, content_type: str, ext_override: str = None) -> tuple[str, str]:
        """保存文件到本地，返回(文件路径, 文件哈希)"""
        file_hash = self._compute_hash(content)

        # 根据类型确定存储子目录
        type_dir_map = {"html": "html", "pdf": "pdf", "image": "images", "doc": "doc", "other": "other"}
        subdir = type_dir_map.get(content_type, "other")

        # 生成文件名：hash + 扩展名
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if ext_override:
            ext = ext_override
        elif "." in Path(path).name:
            ext = Path(path).suffix
        else:
            ext = ".html" if content_type == "html" else ".bin"

        filename = f"{file_hash}{ext}"
        file_path = RAW_DATA_DIR / subdir / filename

        # 避免重复写入
        if not file_path.exists():
            file_path.write_bytes(content)
            logger.debug(f"保存文件: {file_path}")

        return str(file_path), file_hash

    def _fetch(self, url: str, stream: bool = False) -> Optional[requests.Response]:
        """发送HTTP请求"""
        try:
            resp = self.session.get(url, timeout=30, allow_redirects=True, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning(f"请求失败: {url} - {e}")
            return None

    def _detect_content_type_from_response(self, resp: requests.Response) -> str:
        """根据响应的Content-Type头判断实际类型"""
        content_type_header = resp.headers.get("Content-Type", "").lower()

        if "application/pdf" in content_type_header:
            return "pdf"
        for office_type in self.OFFICE_CONTENT_TYPES:
            if office_type.lower() in content_type_header:
                return "doc"
        if any(img_type in content_type_header for img_type in ["image/"]):
            return "image"

        return "html"

    def _clean_body_text(self, text: str, title: str = "") -> str:
        """清理正文文本，去除噪声内容
        
        去除内容:
        - 重复的标题（与页面标题相同的段落）
        - 编辑/审核/浏览次数等元信息
        - 上一篇/下一篇导航
        - 多余空白行
        """
        if not text:
            return ""
        
        lines = text.split("\n")
        cleaned = []
        title_stripped = title.strip()
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行（后面统一处理）
            if not stripped:
                continue
            
            # 跳过与标题重复的行
            if title_stripped and stripped == title_stripped:
                continue
            
            # 跳过编辑/审核/浏览等元信息行
            meta_patterns = [
                r"^编辑[:：]", r"^审核[:：]", r"^浏览[:：]", r"^点击[:：]",
                r"^发布者[:：]", r"^作者[:：]",
                r"^\d{4}-\d{2}-\d{2}\s*$",  # 独立日期行
                r"^日期[:：]", r"^来源[:：]",
            ]
            if any(re.match(p, stripped) for p in meta_patterns):
                continue
            
            # 跳过 上一篇/下一篇 导航
            if re.match(r"^(上一篇|下一篇|上一条|下一条)[：:]", stripped):
                continue
            
            cleaned.append(stripped)
        
        # 合并并去除多余空行
        result = "\n".join(cleaned)
        # 去除连续多个空行
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _extract_detail_page_info(self, url: str, soup: BeautifulSoup) -> dict:
        """从详情页提取结构化信息（标题、正文、发布日期、附件）
        
        只提取正文核心内容，不含导航、侧栏、页脚等噪声
        """
        info = {"title": "", "body_text": "", "publish_date": "", "attachments": [], "author": ""}

        # 提取标题 - 尝试多种选择器（优先匹配高校CMS常见结构）
        title_selectors = [
            ("h2.tit", {}),           # 中国矿业大学详情页标题
            ("h1.content-bt", {}),    # 教务部子站标题
            ("h1", {}),
            ("h2.article-title", {}),
            ("div.article-title", {}),
            ("div.tit", {}),
            ("h2", {}),
            (".article_title", {}),
            (".arti_title", {}),
            (".title", {}),
        ]
        for selector, kwargs in title_selectors:
            title_elem = soup.select_one(selector, **kwargs) if "." in selector else soup.find(selector, **kwargs)
            if title_elem:
                info["title"] = title_elem.get_text(strip=True)
                if info["title"]:
                    break

        # 如果标题还是空，用 <title> 标签
        if not info["title"]:
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                # 去掉 "- 中国矿业大学" 等后缀
                info["title"] = re.sub(r"[-–—]\s*中国矿业大学\s*$", "", title_text).strip()

        # 提取发布日期 - 尝试多种选择器
        date_selectors = [
            ("p.conttime", {}),       # 中国矿业大学详情页日期
            ("h3.content-bt-xia", {}),  # 教务部子站日期
            ("span.date", {}),
            ("span.time", {}),
            ("div.date", {}),
            ("p.date", {}),
            (".arti_metas", {}),
            (".article-info", {}),
            (".pub-date", {}),
            (".post-date", {}),
        ]
        for selector, kwargs in date_selectors:
            date_elem = soup.select_one(selector, **kwargs) if "." in selector else soup.find(selector, **kwargs)
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # 尝试提取日期
                date_match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)", date_text)
                if date_match:
                    info["publish_date"] = date_match.group(1)
                    break

        # 也从meta标签提取
        if not info["publish_date"]:
            meta_date = soup.find("meta", attrs={"name": re.compile(r"date|time|pub", re.I)})
            if meta_date:
                info["publish_date"] = meta_date.get("content", "")

        # 提取正文内容（优先匹配高校CMS常见结构）
        # 关键：只从正文容器中提取，避免包含导航、侧栏等噪声
        body_elem = None
        body_selectors = [
            "div#vsb_content",       # 中国矿业大学详情页正文
            "div.v_news_content",    # 中国矿业大学正文内层
            "section.content_content",  # 教务部子站正文
            "div.wp_articlecontent",    # 主站CMS正文容器
            "div.article-content",
            "div.arti_content",
            "div.conttz",            # 中国矿业大学通知正文class
            "div.content",
            "div.detail-content",
            "div.wp_content",
            "div#article",
            "div.article_body",
            "div.news_content",
            "div.text",
        ]
        for selector in body_selectors:
            body_elem = soup.select_one(selector)
            if body_elem:
                text = body_elem.get_text(separator="\n", strip=True)
                if len(text) > 50:  # 确保有实质内容
                    break
                body_elem = None  # 内容太少，继续尝试下一个选择器
        
        # 如果上述选择器都没匹配到，尝试 ID 匹配 vsb_content_数字（教务部等子站格式）
        if not body_elem:
            for elem in soup.find_all("div", id=re.compile(r"^vsb_content")):
                text = elem.get_text(separator="\n", strip=True)
                if len(text) > 20:
                    body_elem = elem
                    break
        
        if body_elem:
            raw_text = body_elem.get_text(separator="\n", strip=True)
            info["body_text"] = self._clean_body_text(raw_text, info["title"])

        # 提取附件链接（只从正文区域提取，避免导航中的链接干扰）
        search_scope = body_elem if body_elem else soup
        attachment_selectors = [
            "a[href$='.pdf']",
            "a[href$='.doc']",
            "a[href$='.docx']",
            "a[href$='.xls']",
            "a[href$='.xlsx']",
            "a[href$='.ppt']",
            "a[href$='.pptx']",
        ]
        seen_urls = set()
        for selector in attachment_selectors:
            for a_tag in search_scope.select(selector):
                href = a_tag.get("href", "").strip()
                if href and href not in seen_urls:
                    full_url = urljoin(url, href)
                    link_text = a_tag.get_text(strip=True) or Path(urlparse(full_url).path).name
                    info["attachments"].append({
                        "url": full_url,
                        "text": link_text,
                        "type": self._classify_url(full_url),
                    })
                    seen_urls.add(href)

        # 从 iframe 中提取嵌入的PDF（部分学术报告正文通过iframe嵌入PDF）
        if body_elem:
            for iframe in body_elem.find_all("iframe"):
                src = iframe.get("src", "").strip()
                if src and src.lower().endswith(".pdf"):
                    full_url = urljoin(url, src)
                    if full_url not in seen_urls:
                        info["attachments"].append({
                            "url": full_url,
                            "text": Path(urlparse(full_url).path).name,
                            "type": "pdf",
                        })
                        seen_urls.add(full_url)

        # 提取作者/来源
        author_selectors = [
            ("span.author", {}),
            ("span.source", {}),
            (".arti_author", {}),
            (".article-source", {}),
        ]
        for selector, kwargs in author_selectors:
            author_elem = soup.select_one(selector, **kwargs) if "." in selector else soup.find(selector, **kwargs)
            if author_elem:
                info["author"] = author_elem.get_text(strip=True)
                break

        return info

    def _is_detail_page(self, url: str, soup: BeautifulSoup) -> bool:
        """判断是否为详情页（含正文内容的页面，而非列表/导航页）
        
        判断依据（满足任一即为详情页）:
        1. URL包含 /info/ 路径（高校CMS标准格式）
        2. URL以数字.htm/html结尾（如 73287.htm）
        3. URL包含 /content.jsp 且有文章标题
        4. URL包含 /page.htm（信息公开等子站的文章格式，如 /14/a3/c2031a595107/page.htm）
        5. URL是具体文章页（非list.htm/main.htm/index.htm等列表页）
        6. 页面中存在正文容器（vsb_content等）
        """
        path = urlparse(url).path.lower()
        
        # 规则1: /info/ 路径
        if "/info/" in path:
            return True
        
        # 规则2: URL以数字ID.htm结尾（如 73287.htm, 1623_1.htm）
        if re.search(r"/\d+(_\d+)?\.htm[l]?$", path):
            return True
        
        # 规则3: JSP详情页（如 content.jsp?wbnewsid=xxx）
        if "content.jsp" in path and "wbnewsid" in url.lower():
            return True
        
        # 规则4: /page.htm 格式（信息公开等子站的文章格式）
        if path.endswith("/page.htm"):
            return True
        
        # 规则5: 排除明确的列表页/导航页，其余.htm结尾视为详情页
        list_page_patterns = ["/list.htm", "/main.htm", "/index.htm", "/index.html",
                              "/index.jsp", "list.jsp"]
        if path.endswith((".htm", ".html")) and not any(p in path for p in list_page_patterns):
            # 进一步检查：URL路径深度>=2 且 不是常见列表页格式
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2 and not parts[-1].startswith("list"):
                return True
        
        # 规则6: 页面中存在正文容器
        body_container_selectors = [
            "div#vsb_content",
            "div.v_news_content",
            "section.content_content",
            "div.article-content",
            "div.conttz",
        ]
        for selector in body_container_selectors:
            if soup.select_one(selector):
                text = soup.select_one(selector).get_text(strip=True)
                if len(text) > 100:
                    return True
        
        return False

    def _parse_html(self, url: str, resp: requests.Response) -> CrawlResult:
        """解析HTML页面，提取标题、正文和链接"""
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        # 智能判断是否是详情页
        is_detail = self._is_detail_page(url, soup)

        # 提取标题
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 保存原始HTML
        file_path, file_hash = self._save_file(url, resp.content, "html")

        # 提取页面中的所有链接
        links = []
        pdf_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue
            full_url = urljoin(url, href)
            links.append(full_url)
            # 收集PDF链接
            if self._classify_url(full_url) == "pdf":
                link_text = a_tag.get_text(strip=True) or ""
                pdf_links.append({"url": full_url, "text": link_text})

        # 详情页提取更丰富的信息；列表页不提取body_text（只有链接列表，无实质内容）
        detail_info = {}
        if is_detail:
            detail_info = self._extract_detail_page_info(url, soup)
            if detail_info.get("title"):
                title = detail_info["title"]
        else:
            # 列表页：尝试提取简要描述（如首页的公告摘要）
            # 不提取完整body_text，避免噪声数据入库
            pass

        metadata = {
            "links_found": len(links),
            "page_links": links,
            "is_detail_page": is_detail,
            "source_site": self.name,
            "content_length": len(resp.content),
        }

        # 合并详情信息
        if detail_info:
            metadata.update({
                "body_text": detail_info.get("body_text", ""),
                "publish_date": detail_info.get("publish_date", ""),
                "author": detail_info.get("author", ""),
                "attachments": detail_info.get("attachments", []),
            })

        # 如果发现PDF附件，加入下载队列
        if pdf_links:
            metadata["pdf_links_found"] = len(pdf_links)
            for pdf_info in pdf_links:
                pdf_url = pdf_info["url"]
                if pdf_url not in self.visited_urls and self._is_allowed_url(pdf_url):
                    self.queue.append((pdf_url, 0))  # PDF文件不增加深度
                    logger.info(f"发现PDF附件: {pdf_info['text'] or pdf_url}")

        return CrawlResult(
            url=url,
            title=title,
            content_type="html",
            file_path=file_path,
            file_hash=file_hash,
            status_code=resp.status_code,
            metadata=metadata,
        )

    def _download_file(self, url: str, content_type: str) -> CrawlResult:
        """下载非HTML文件(PDF/图片/Office文档)"""
        resp = self._fetch(url)
        if not resp:
            return CrawlResult(
                url=url, title="", content_type=content_type,
                status_code=0, error="下载失败",
            )

        # 根据响应Content-Type校正文件类型
        actual_type = self._detect_content_type_from_response(resp)
        if actual_type != "html" and actual_type != content_type:
            logger.info(f"URL类型校正: {url} | 预期={content_type} | 实际={actual_type}")
            content_type = actual_type

        ext_map = {"pdf": ".pdf", "doc": ".doc", "image": ""}
        ext = ext_map.get(content_type)
        file_path, file_hash = self._save_file(url, resp.content, content_type, ext_override=ext)

        # 提取文件名作为标题
        parsed = urlparse(url)
        title = Path(parsed.path).name or content_type.upper()

        return CrawlResult(
            url=url,
            title=title,
            content_type=content_type,
            file_path=file_path,
            file_hash=file_hash,
            status_code=resp.status_code,
            metadata={
                "source_site": self.name,
                "content_length": len(resp.content),
                "original_filename": Path(parsed.path).name,
            },
        )

    def crawl(self, max_pages: int = 50) -> list[CrawlResult]:
        """执行爬取

        Args:
            max_pages: 最大爬取页面数

        Returns:
            爬取结果列表
        """
        logger.info(f"开始爬取站点: {self.name} (base: {self.base_url})")

        # 初始化队列
        for url in self.start_urls:
            if url not in self.visited_urls:
                self.queue.append((url, 0))

        pages_crawled = 0

        while self.queue and pages_crawled < max_pages:
            url, current_depth = self.queue.pop(0)

            # 跳过已访问的URL
            if url in self.visited_urls:
                continue

            # 检查URL是否在允许范围内
            if not self._is_allowed_url(url):
                logger.debug(f"URL不在允许范围: {url}")
                continue

            self.visited_urls.add(url)
            logger.info(f"爬取 [{pages_crawled + 1}/{max_pages}] depth={current_depth}: {url}")

            # 根据内容类型处理
            content_type = self._classify_url(url)

            if content_type == "html":
                resp = self._fetch(url)
                if not resp:
                    self.results.append(CrawlResult(
                        url=url, title="", content_type="html",
                        status_code=0, error="请求失败",
                    ))
                    continue

                result = self._parse_html(url, resp)
                self.results.append(result)
                pages_crawled += 1

                # 提取子链接加入队列（详情页优先）
                if current_depth < self.depth and result.metadata.get("page_links"):
                    detail_links = []
                    other_links = []
                    for link in result.metadata["page_links"]:
                        if link not in self.visited_urls and self._is_allowed_url(link):
                            # 识别详情页链接
                            link_path = urlparse(link).path.lower()
                            is_detail_link = (
                                "/info/" in link_path or
                                re.search(r"/\d+(_\d+)?\.htm[l]?$", link_path) or
                                "content.jsp" in link_path or
                                link_path.endswith("/page.htm") or
                                (link_path.endswith((".htm", ".html")) and
                                 not any(p in link_path for p in ["/list.htm", "/main.htm", "/index.htm"]))
                            )
                            if is_detail_link:
                                detail_links.append((link, current_depth + 1))
                            else:
                                other_links.append((link, current_depth + 1))
                    # 详情页优先加入队列头部
                    self.queue = detail_links + self.queue + other_links
            else:
                # 下载非HTML文件
                result = self._download_file(url, content_type)
                self.results.append(result)
                pages_crawled += 1

            # 请求间隔
            time.sleep(self.delay)

        logger.info(
            f"爬取完成: {self.name} | "
            f"总页面={pages_crawled} | "
            f"成功={sum(1 for r in self.results if not r.error)} | "
            f"失败={sum(1 for r in self.results if r.error)}"
        )
        return self.results


def load_sites_config() -> list[dict]:
    """从sites.yaml加载站点配置"""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config" / "sites.yaml"
    if not config_path.exists():
        logger.error(f"站点配置文件不存在: {config_path}")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("sites", [])


def run_crawl(max_pages: int = 50, site_names: Optional[list[str]] = None) -> list[CrawlResult]:
    """执行爬取任务

    Args:
        max_pages: 每个站点最大爬取页面数
        site_names: 指定爬取的站点名称列表，None表示全部

    Returns:
        所有爬取结果
    """
    all_results: list[CrawlResult] = []
    sites = load_sites_config()

    if site_names:
        sites = [s for s in sites if s["name"] in site_names]

    if not sites:
        logger.error("没有找到可爬取的站点配置")
        return all_results

    for site in sites:
        crawler = SchoolCrawler(site)
        results = crawler.crawl(max_pages=max_pages)
        all_results.extend(results)

    # 打印摘要
    total = len(all_results)
    by_type = {}
    for r in all_results:
        by_type[r.content_type] = by_type.get(r.content_type, 0) + 1

    # 统计附件数量
    total_attachments = sum(
        len(r.metadata.get("attachments", []))
        for r in all_results
        if r.metadata.get("attachments")
    )
    detail_pages = sum(1 for r in all_results if r.metadata.get("is_detail_page"))

    logger.info(
        f"爬取摘要: 总计={total}, 按类型={by_type}, "
        f"详情页={detail_pages}, 附件={total_attachments}"
    )

    return all_results
