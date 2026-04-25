"""FastAPI server wiring for AgentBoundary-v1."""

try:
    from openenv.core.env_server.http_server import create_app
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


app = create_app(
    AgentBoundaryEnvironment,
    AgentBoundaryAction,
    AgentBoundaryObservation,
    env_name="agentboundary-v1",
    max_concurrent_envs=4,
)


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
