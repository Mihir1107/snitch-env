"""FastAPI + WebSocket server implementing the OpenEnv contract for SnitchEnv."""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from env.snitch_env import SnitchEnv

app = FastAPI(title="snitch-env", version="0.1.0")
_env = SnitchEnv()


class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    obs, info = _env.reset(seed=req.seed)
    return {"observation": obs, "info": info}


@app.post("/step")
def step(req: StepRequest) -> dict:
    obs, reward, done, info = _env.step(req.action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            kind = msg.get("type")
            if kind == "reset":
                obs, info = _env.reset(seed=msg.get("seed"))
                await websocket.send_json({"type": "reset", "observation": obs, "info": info})
            elif kind == "step":
                obs, reward, done, info = _env.step(msg.get("action"))
                await websocket.send_json(
                    {"type": "step", "observation": obs, "reward": reward, "done": done, "info": info}
                )
            else:
                await websocket.send_json({"type": "error", "message": f"unknown message: {kind}"})
    except WebSocketDisconnect:
        return
