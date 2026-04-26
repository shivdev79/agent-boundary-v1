"""FastAPI server wiring for AgentBoundary-v1."""

import asyncio
from typing import Any, Dict

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.serialization import serialize_observation
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AgentBoundaryAction, AgentBoundaryObservation
    from .agentv1_environment import AgentBoundaryEnvironment
except ImportError:  # pragma: no cover
    from models import AgentBoundaryAction, AgentBoundaryObservation
    from server.agentv1_environment import AgentBoundaryEnvironment

from fastapi import Body, HTTPException, status
from fastapi.routing import APIRoute

app = create_app(
    AgentBoundaryEnvironment,
    AgentBoundaryAction,
    AgentBoundaryObservation,
    env_name="agentboundary-v1",
    max_concurrent_envs=4,
)

# ── Stateful HTTP session ─────────────────────────────────────────────────────
# The OpenEnv framework's /reset and /step HTTP endpoints are stateless by
# design — each request creates a fresh environment instance, uses it, then
# discards it. This means sequential reset → step calls over HTTP hit different
# instances and produce wrong results. We replace those routes with a single
# persistent environment so the HTTP REST API and direct curl testing work.
#
# The WebSocket /ws endpoint (used by the playground UI) is unaffected — it
# already creates a dedicated environment per connection and is fully stateful.

_http_env = AgentBoundaryEnvironment()
_http_lock = asyncio.Lock()

# Remove the framework's stateless /reset and /step routes before registering ours
app.router.routes = [
    r for r in app.router.routes
    if not (isinstance(r, APIRoute) and r.path in {"/reset", "/step"})
]


@app.post("/reset", tags=["Environment Control"], summary="Reset the environment")
async def stateful_reset(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    async with _http_lock:
        task_id = body.get("task_id")
        seed = body.get("seed")
        obs = _http_env.reset(task_id=task_id, seed=seed)
        serialized = serialize_observation(obs)
        return {"observation": serialized["observation"], "reward": None, "done": False}


@app.post("/step", tags=["Environment Control"], summary="Execute an action")
async def stateful_step(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    action_data = body.get("action")
    if action_data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": "Field required", "loc": ["body", "action"]}],
        )
    try:
        action = AgentBoundaryAction(**action_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    async with _http_lock:
        obs = _http_env.step(action)
        serialized = serialize_observation(obs)
        return {
            "observation": serialized["observation"],
            "reward": obs.reward,
            "done": obs.done,
        }


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the local uvicorn server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Entry point used by validators and project scripts."""
    run_server()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
