"""解析启动脚本 - 将爬虫结果解析为结构化文档"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from parser.router import ParserRouter


def main():
    parser = argparse.ArgumentParser(description="文档解析器")
    parser.add_argument("--input", type=str, default="data/crawl_results.json",
                        help="爬虫结果JSON文件路径")
    parser.add_argument("--output", type=str, default="data/parsed_docs_current.json",
                        help="解析结果输出路径")
    parser.add_argument("--all", action="store_true",
                        help="解析所有页面（默认只解析详情页）")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    # 配置日志
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    # 初始化路由器
    router = ParserRouter()

    # 批量解析
    logger.info(f"开始解析: {args.input}")
    docs = router.parse_crawl_results(
        json_path=args.input,
        only_detail=not args.all,
    )

    if not docs:
        logger.warning("未解析到任何文档")
        return

    # 统计
    by_type = {}
    by_site = {}
    total_chars = 0
    for doc in docs:
        ct = doc.content_type.value
        by_type[ct] = by_type.get(ct, 0) + 1
        by_site[doc.source_site] = by_site.get(doc.source_site, 0) + 1
        total_chars += len(doc.text)

    print("\n" + "=" * 60)
    print("解析结果摘要")
    print("=" * 60)
    print(f"\n文档总数: {len(docs)}")
    print(f"总字符数: {total_chars:,}")
    print(f"\n按类型统计:")
    for ct, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")
    print(f"\n按站点统计:")
    for site, count in sorted(by_site.items(), key=lambda x: -x[1]):
        print(f"  {site}: {count}")

    # 展示部分结果
    print(f"\n--- 解析结果示例 (最多5条) ---\n")
    for doc in docs[:5]:
        print(f"[{doc.content_type.value}] {doc.title[:50]}")
        print(f"  来源: {doc.source_site} | URL: {doc.source_url[:60]}")
        print(f"  正文长度: {len(doc.text)} 字符")
        print(f"  正文预览: {doc.text[:100].replace(chr(10), ' ')}...")
        print()

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([doc.to_dict() for doc in docs], f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
