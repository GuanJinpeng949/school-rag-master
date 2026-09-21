"""爬取启动脚本 - 支持命令行运行爬虫并查看结果"""
import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import ensure_dirs
from crawler.spider import run_crawl, CrawlResult


def print_results_summary(results: list[CrawlResult]):
    """打印爬取结果摘要"""
    print("\n" + "=" * 80)
    print("爬取结果摘要")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if not r.error)
    failed = sum(1 for r in results if r.error)

    print(f"\n总页面数: {total} | 成功: {success} | 失败: {failed}")

    # 按类型统计
    by_type = {}
    for r in results:
        by_type[r.content_type] = by_type.get(r.content_type, 0) + 1
    print(f"按类型: {by_type}")

    # 详情页和附件统计
    detail_pages = sum(1 for r in results if r.metadata.get("is_detail_page"))
    total_attachments = sum(
        len(r.metadata.get("attachments", []))
        for r in results
        if r.metadata.get("attachments")
    )
    print(f"详情页: {detail_pages} | 附件: {total_attachments}")

    # 展示部分结果详情
    print(f"\n--- 详细结果 (最多显示25条) ---\n")
    for i, r in enumerate(results[:25]):
        status = "OK" if not r.error else "FAIL"
        is_detail = "[D]" if r.metadata.get("is_detail_page") else "   "
        print(f"[{i+1:2d}] {status} {is_detail} [{r.content_type.upper():4s}] {r.title[:55] if r.title else '(无标题)'}")
        print(f"      URL: {r.url}")
        if r.file_path:
            print(f"      File: {r.file_path}")
        if r.error:
            print(f"      Error: {r.error}")
        if r.metadata.get("publish_date"):
            print(f"      Date: {r.metadata['publish_date']}")
        if r.metadata.get("attachments"):
            for att in r.metadata["attachments"]:
                print(f"      Attachment: [{att['type']}] {att['text'][:50]}")
        if r.metadata.get("body_text"):
            body_preview = r.metadata["body_text"][:80].replace("\n", " ")
            print(f"      Body: {body_preview}...")
        print()


def save_results_json(results: list[CrawlResult], output_path: str):
    """保存结果为JSON文件"""
    data = []
    for r in results:
        data.append({
            "url": r.url,
            "title": r.title,
            "content_type": r.content_type,
            "file_path": r.file_path,
            "file_hash": r.file_hash,
            "status_code": r.status_code,
            "error": r.error,
            "metadata": r.metadata,
            "crawled_at": r.crawled_at,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="学校官网RAG爬虫")
    parser.add_argument("--max-pages", type=int, default=200, help="每个站点最大爬取页面数")
    parser.add_argument("--sites", nargs="*", default=None, help="指定爬取站点名称(空格分隔)")
    parser.add_argument("--output", type=str, default="data/crawl_results.json", help="结果输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    # 配置日志
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    # 确保目录
    ensure_dirs()

    # 执行爬取
    logger.info("开始爬取任务...")
    results = run_crawl(max_pages=args.max_pages, site_names=args.sites)

    # 打印摘要
    print_results_summary(results)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_json(results, str(output_path))


if __name__ == "__main__":
    main()
