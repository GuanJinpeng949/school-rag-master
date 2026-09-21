"""图片OCR解析器 - 从图片中识别文字

核心功能:
1. 使用PaddleOCR识别图片中的中英文文字
2. 支持常见图片格式：JPG/PNG/GIF/BMP/WEBP
3. 支持角度检测和旋转校正

设计说明:
- 学校官网中部分内容以图片形式发布（如公告截图、通知图片）
- 爬虫阶段保存了图片文件，解析器负责OCR文字提取
- PaddleOCR中文识别能力强，适合国内高校官网场景
"""
from pathlib import Path
from typing import Optional

from loguru import logger

from parser.base import BaseParser, ContentType, ParsedDocument


class ImageParser(BaseParser):
    """图片OCR解析器"""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

    @property
    def supported_types(self) -> list[ContentType]:
        return [ContentType.IMAGE]

    def parse(self, file_path: str | Path, metadata: Optional[dict] = None) -> list[ParsedDocument]:
        """解析图片文件，使用OCR提取文字"""
        file_path = Path(file_path)
        metadata = metadata or {}

        if not file_path.exists():
            logger.warning(f"图片文件不存在: {file_path}")
            return []

        # 检查扩展名
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的图片格式: {file_path.suffix}")
            return []

        source_url = metadata.get("url", "")
        source_site = metadata.get("source_site", "")
        file_hash = metadata.get("file_hash", file_path.stem[:16])
        title = metadata.get("title", file_path.stem)

        # OCR提取
        text = self._ocr_extract(file_path)
        if not text:
            logger.debug(f"图片OCR结果为空: {file_path.name}")
            return []

        doc_id = self._generate_doc_id(source_url, file_hash)

        doc = ParsedDocument(
            doc_id=doc_id,
            text=text,
            source_url=source_url,
            source_site=source_site,
            title=title,
            content_type=ContentType.IMAGE,
            publish_date=metadata.get("publish_date", ""),
            author=metadata.get("author", ""),
            file_path=str(file_path),
            file_hash=file_hash,
            chunk_index=0,
            total_chunks=1,
            attachments=metadata.get("attachments", []),
            extra={"ocr_engine": "paddleocr"},
        )

        return [doc]

    def _ocr_extract(self, file_path: Path) -> str:
        """使用PaddleOCR提取图片中的文字"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR未安装，无法进行图片OCR")
            return ""

        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            result = ocr.ocr(str(file_path), cls=True)

            if not result or not result[0]:
                return ""

            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.5:  # 置信度过滤
                        texts.append(text)

            return "\n".join(texts)

        except Exception as e:
            logger.error(f"图片OCR提取失败: {file_path} - {e}")
            return ""
