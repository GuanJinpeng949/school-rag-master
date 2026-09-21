"""FastAPI应用入口 - CORS、路由注册、生命周期

启动方式:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from loguru import logger

from config.settings import settings, ensure_dirs


# 前端构建产物路径
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动和关闭"""
    # 启动
    logger.info("校园智搜 API 启动中...")
    ensure_dirs()
    if FRONTEND_DIST.exists():
        logger.info(f"前端静态文件: {FRONTEND_DIST}")
    logger.info(f"API地址: http://{settings.api.api_host}:{settings.api.api_port}")
    logger.info("API启动完成")

    yield

    # 关闭
    logger.info("API关闭中...")
    from api.dependencies import _metadata_store
    if _metadata_store:
        _metadata_store.close()
    logger.info("API已关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="校园智搜 API",
        description="学校官网RAG智能检索系统后端API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS - 允许前端跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册API路由（优先级高于静态文件）
    from api.routes.search import router as search_router
    from api.routes.system import router as system_router
    app.include_router(search_router)
    app.include_router(system_router)

    # 前端静态文件挂载
    # 关键：使用 /static 挂载静态资源（JS/CSS），用 catch-all 路由处理页面
    if FRONTEND_DIST.exists():
        # 挂载 /assets 目录（Vite构建的JS/CSS等）
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static-assets")

        # 其他静态资源（favicon等）
        for static_file in ["vite.svg"]:
            fp = FRONTEND_DIST / static_file
            if fp.exists():

                @app.get(f"/{static_file}", include_in_schema=False)
                async def _static_file(path: str = static_file):
                    return FileResponse(str(FRONTEND_DIST / path))

        # SPA catch-all：所有非API请求返回 index.html
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_spa(request: Request, path: str):
            """SPA回退路由：先尝试静态文件，否则返回index.html"""
            # 尝试找对应的静态文件
            file_path = FRONTEND_DIST / path
            if path and file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # SPA回退：返回index.html让前端路由处理
            index_html = FRONTEND_DIST / "index.html"
            if index_html.exists():
                return FileResponse(str(index_html))
            return HTMLResponse("<h1>Frontend not built</h1><p>Run: cd frontend && npm run build</p>", status_code=404)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=settings.api.api_reload,
    )
