from __future__ import annotations
import asyncio
import os
import threading
from protobuf2openai.app import app as openai_server  # FastAPI app
from server import create_app, startup_tasks
import uvicorn


async def main():
    # 在后台线程启动 warp server
    warp_app = create_app()
    await startup_tasks()
    
    # 启动 warp server 的后台线程
    warp_thread = threading.Thread(
        target=uvicorn.run,
        args=(warp_app,),
        kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "info", "access_log": True},
        daemon=True
    )
    warp_thread.start()
    
    try:
        from warp2protobuf.core.auth import refresh_jwt_if_needed as _refresh_jwt
        await _refresh_jwt()
    except Exception:
        pass
    

if __name__ == "__main__":
    asyncio.run(main())
    uvicorn.run(
        openai_server,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8010")),
        log_level="info",
    )