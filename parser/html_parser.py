"""HTML解析器 - 从原始HTML文件中提取正文内容

核心功能:
1. 正文提取：优先匹配高校CMS常见结构选择器，兜底用readability-lxml
2. 表格结构化：将HTML表格转为Markdown格式保留结构
3. 元数据提取：标题、日期、作者、附件
4. 正文去噪：去除导航栏、页脚、JS脚本等噪声

设计说明:
- 爬虫阶段已部分提取body_text，但覆盖率仅63%，解析器负责完整提取
- 对于爬虫已有body_text的页面，解析器可以重新提取以获得更干净的结果
- 支持从原始HTML文件重新解析，不依赖爬虫阶段的提取结果
"""
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from loguru import logger

from parser.base import BaseParser, ContentType, ParsedDocument


class HTMLParser(BaseParser):
    """HTML文档解析器"""

    # 正文容器选择器（按优先级排列）
    BODY_SELECTORS = [
        "div#vsb_content",          # 中国矿业大学详情页正文
        "div.v_news_content",       # 中国矿业大学正文内层
        "section.content_content",  # 教务部子站正文
        "div.wp_articlecontent",    # 主站CMS正文容器
        "div.article-content",
        "div.arti_content",
        "div.conttz",               # 通知正文
        "div.wp_content",
        "div#article",
        "div.article_body",
        "div.news_content",
        "div.text",
        "div.content",
        "div.detail-content",
    ]

    # 标题选择器
    TITLE_SELECTORS = [
        "h2.tit",
        "h1.content-bt",
        "h1",
        "h2.article-title",
        "div.article-title",
        "div.tit",
        "h2",
        ".article_title",
        ".arti_title",
        ".title",
    ]

    # 日期选择器
    DATE_SELECTORS = [
        "p.conttime",
        "h3.content-bt-xia",
        "span.date",
        "span.time",
        "div.date",
        "p.date",
        ".arti_metas",
        ".article-info",
        ".pub-date",
        ".post-date",
    ]

    # 作者/来源选择器
    AUTHOR_SELECTORS = [
        "span.author",
        "span.source",
        ".arti_author",
        ".article-source",
    ]

    # 需要移除的噪声标签
    NOISE_TAGS = ["script", "style", "nav", "footer", "header", "iframe", "noscript"]

    # 需要移除的噪声CSS类（匹配即可移除）
    NOISE_CLASSES = [
        "nav", "navbar", "navigation", "menu", "sidebar", "side-bar",
        "footer", "header", "banner", "advertisement", "ad-", "breadcrumb",
        "pagination", "pager", "page-nav", "copyright", "copyright-bar",
        "share", "social", "comment", "related-posts", "recommend",
    ]

    @property
    def supported_types(self) -> list[ContentType]:
        return [ContentType.HTML]

    def parse(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """解析HTML文件，提取正文和元数据"""
        file_path = Path(file_path)
        metadata = metadata or {}

        if not file_path.exists():
            logger.warning(f"HTML文件不存在: {file_path}")
            return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"读取HTML文件失败: {file_path} - {e}")
            return []

        # 把 content 中的原始 HTML 字符串解析成一个可以查询和操作的 HTML 文档树，使字符串content可以被解析器使用
        soup = BeautifulSoup(content, "lxml")

        # 如果metadata中有url，优先使用；否则从文件路径反推
        source_url = metadata.get("url", "")
        source_site = metadata.get("source_site", "")
        file_hash = metadata.get("file_hash", file_path.stem[:16])

        # 提取元数据
        title = self._extract_title(soup, metadata)
        publish_date = self._extract_date(soup, metadata)
        author = self._extract_author(soup, metadata)
        attachments = self._extract_attachments(soup, metadata, source_url)

        # 提取正文
        body_text = self._extract_body(soup, title)
        if not body_text:
            # 兜底：使用 readability-lxml 提取
            body_text = self._extract_with_readability(content)
        if not body_text:
            logger.debug(f"HTML正文为空: {file_path.name}")
            return []

        # 处理正文中的表格（转为Markdown格式）
        body_elem = self._find_body_element(soup)
        if body_elem:
            body_text = self._process_tables(body_elem, body_text)

        doc_id = self._generate_doc_id(source_url, file_hash)

        doc = ParsedDocument(
            doc_id=doc_id,
            text=body_text,
            source_url=source_url,
            source_site=source_site,
            title=title,
            content_type=ContentType.HTML,
            publish_date=publish_date,
            author=author,
            file_path=str(file_path),
            file_hash=file_hash,
            chunk_index=0,
            total_chunks=1,
            attachments=attachments,
        )

        return [doc]

    def _extract_title(self, soup: BeautifulSoup, metadata: dict) -> str:
        """提取标题"""
        # 优先使用metadata中已有的标题
        if metadata.get("title"):
            return metadata["title"]

        for selector in self.TITLE_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                if title:
                    return title

        # 兜底：从 <title> 标签提取
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # 去掉 "- 中国矿业大学" 等后缀
            return re.sub(r"[-–—]\s*中国矿业大学\s*$", "", title).strip()

        return ""

    def _extract_date(self, soup: BeautifulSoup, metadata: dict) -> str:
        """提取发布日期"""
        if metadata.get("publish_date"):
            return metadata["publish_date"]

        for selector in self.DATE_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                date_text = elem.get_text(strip=True)
                match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)", date_text)
                if match:
                    return match.group(1)

        # 从meta标签提取
        meta_date = soup.find("meta", attrs={"name": re.compile(r"date|time|pub", re.I)})
        if meta_date:
            return meta_date.get("content", "")

        return ""

    def _extract_author(self, soup: BeautifulSoup, metadata: dict) -> str:
        """提取作者/来源"""
        if metadata.get("author"):
            return metadata["author"]

        for selector in self.AUTHOR_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text(strip=True)

        return ""

    def _extract_attachments(self, soup: BeautifulSoup, metadata: dict, source_url: str) -> list[dict]:
        """提取附件链接"""
        # 优先使用爬虫阶段已提取的附件信息
        if metadata.get("attachments"):
            return metadata["attachments"]

        attachments = []
        body_elem = self._find_body_element(soup)
        search_scope = body_elem if body_elem else soup

        attachment_exts = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]
        seen = set()

        # 找search_scope所有带有 href 属性的 <a> 标签
        for a_tag in search_scope.find_all("a", href=True):
            # 读取链接地址
            href = a_tag["href"].strip()
            # 跳过无效链接
            if not href or href.startswith("javascript:") or href.startswith("#"):
                continue
            # 只要 href 以任意一个支持的附件后缀结尾，就进入处理逻辑
            if any(href.lower().endswith(ext) for ext in attachment_exts):
                if href not in seen:
                    full_url = urljoin(source_url, href) if source_url else href
                    link_text = a_tag.get_text(strip=True) or Path(urlparse(full_url).path).name
                    # 判断附件类型
                    ext = Path(urlparse(full_url).path).suffix.lower()
                    att_type = "pdf" if ext == ".pdf" else "doc"
                    attachments.append({
                        "url": full_url,
                        "text": link_text,
                        "type": att_type,
                    })
                    seen.add(href)

        return attachments

    def _find_body_element(self, soup: BeautifulSoup) -> Optional[Tag]:
        """查找正文容器元素"""
        # 固定选择器
        for selector in self.BODY_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if len(text) > 50:
                    return elem

        # 尝试动态 ID vsb_content_数字 格式
        for elem in soup.find_all("div", id=re.compile(r"^vsb_content")):
            text = elem.get_text(strip=True)
            if len(text) > 20:
                return elem

        return None

    def _extract_body(self, soup: BeautifulSoup, title: str = "") -> str:
        """提取正文文本"""
        body_elem = self._find_body_element(soup)
        if not body_elem:
            return ""

        # 先清理噪声标签
        self._remove_noise(body_elem)

        # 提取文本
        text = body_elem.get_text(separator="\n", strip=True)

        # 清理文本
        return self._clean_text(text, title)

    def _remove_noise(self, elem: Tag):
        """从元素中移除噪声标签"""
        for tag_name in self.NOISE_TAGS:
            for tag in elem.find_all(tag_name):
                tag.decompose()

        # 移除噪声class的div，按 CSS 类名删除（需要收集后统一删除，避免迭代中修改）
        to_remove = []
        for div in elem.find_all("div"):
            classes = div.get("class")
            if not classes:
                continue
            # 将单个字符串转换为列表
            if isinstance(classes, str):
                classes = [classes]
            for cls in classes:
                if not cls:
                    continue
                for noise_cls in self.NOISE_CLASSES:
                    if noise_cls in cls.lower():
                        to_remove.append(div)
                        break
        for div in to_remove:
            div.decompose()

    def _clean_text(self, text: str, title: str = "") -> str:
        """清理正文文本，去除噪声内容"""
        if not text:
            return ""

        lines = text.split("\n")
        cleaned = []
        title_stripped = title.strip()

        for line in lines:
            stripped = line.strip()

            # 跳过空行
            if not stripped:
                continue

            # 跳过与标题重复的行
            if title_stripped and stripped == title_stripped:
                continue

            # 跳过元信息行
            meta_patterns = [
                r"^编辑[:：]", r"^审核[:：]", r"^浏览[:：]", r"^点击[:：]",
                r"^发布者[:：]", r"^作者[:：]",
                r"^\d{4}-\d{2}-\d{2}\s*$",
                r"^日期[:：]", r"^来源[:：]",
            ]
            if any(re.match(p, stripped) for p in meta_patterns):
                continue

            # 跳过 上一篇/下一篇 导航
            if re.match(r"^(上一篇|下一篇|上一条|下一条)[：:]", stripped):
                continue

            cleaned.append(stripped)

        # 把剩余内容重新拼接起来，保留3个空行作为段落分隔
        result = "\n".join(cleaned)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _extract_with_readability(self, html_content: str) -> str:
        """当固定正文选择器全部失败时，使用 readability-lxml 兜底提取正文"""
        try:
            from readability import Document
            doc = Document(html_content)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            return self._clean_text(text)
        except Exception as e:
            logger.debug(f"readability提取失败: {e}")
            return ""

    def _process_tables(self, body_elem: Tag, body_text: str) -> str:
        """将正文中的HTML表格转为Markdown格式，融入正文文本"""
        tables = body_elem.find_all("table")
        if not tables:
            return body_text

        # 构建表格的Markdown表示
        table_markers = {}
        for i, table in enumerate(tables):
            md_table = self._table_to_markdown(table)
            if md_table:
                # 生成一个占位标记
                marker = f"__TABLE_{i}__"
                table_markers[marker] = md_table
                # 在原始文本中，表格内容可能已经被提取为纯文本（格式较差）
                # 这里我们不替换原文本，而是在文本末尾追加结构化表格，避免表格原始文本在前面已经被提取后丢失

        if table_markers:
            # 追加结构化表格到正文末尾
            tables_section = "\n\n---\n\n".join(table_markers.values())
            return f"{body_text}\n\n---\n\n【结构化表格】\n\n{tables_section}"

        return body_text

    def _table_to_markdown(self, table: Tag) -> str:
        """将HTML表格转为Markdown格式"""
        rows = []
        # 行遍历
        for tr in table.find_all("tr"):
            cells = []
            # 单元格遍历
            for td in tr.find_all(["td", "th"]):
                # 清理换行和 | 字符
                cell_text = td.get_text(strip=True).replace("\n", " ").replace("|", "｜")
                cells.append(cell_text)
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        # 统一列数
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # 生成Markdown表格
        lines = []
        # 表头（第一行）
        lines.append("| " + " | ".join(rows[0]) + " |")
        # 分隔行
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        # 数据行
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
