import httpx
from typing import Any, Dict, Optional


class PipelineDebugEnvClient:
    """
    Async HTTP client SDK for the pipeline_debug_env OpenEnv server.
    Usage:
        async with PipelineDebugEnvClient("http://localhost:7860") as client:
            obs = await client.reset()
            result = await client.step(action)
            state = await client.state()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_level: Optional[str] = None) -> Dict[str, Any]:
        body = {}
        if task_level:
            body["task_level"] = task_level
        response = await self._client.post("/reset", json=body)
        response.raise_for_status()
        return response.json()

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post("/step", json=action)
        response.raise_for_status()
        return response.json()

    async def state(self) -> Dict[str, Any]:
        response = await self._client.get("/state")
        response.raise_for_status()
        return response.json()
