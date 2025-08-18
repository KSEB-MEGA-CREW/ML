import logging
import asyncio
from collections import deque
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import json
import uuid

from app.models.predictor import SignLanguagePredictor

logger = logging.getLogger(__name__)
router = APIRouter()


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_buffers: Dict[str, deque] = {}
        self.predictor = SignLanguagePredictor()

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_buffers[session_id] = deque(maxlen=10)
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_buffers:
            del self.session_buffers[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

    async def process_frame(self, session_id: str, keypoints: List[float]) -> bool:
        """10프레임 누적 후 예측 수행"""
        if session_id not in self.session_buffers:
            return False

        self.session_buffers[session_id].append(keypoints)

        if len(self.session_buffers[session_id]) == 10:
            try:
                keypoints_sequence = list(self.session_buffers[session_id])
                result = await self.predictor.predict_sequence(keypoints_sequence)

                await self.send_message(result, session_id)
                self.session_buffers[session_id].clear()

                return True

            except Exception as e:
                logger.error(f"Prediction error for session {session_id}: {e}")
                error_message = {
                    "success": False,
                    "error": f"Prediction failed: {str(e)}",
                    "prediction": None,
                }
                await self.send_message(error_message, session_id)
                return False

        return True


manager = WebSocketManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    session_id = str(uuid.uuid4())

    try:
        await manager.connect(websocket, session_id)

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if "keypoints" in message:
                keypoints = message["keypoints"]

                if len(keypoints) != 194:
                    error_msg = {
                        "success": False,
                        "error": f"Invalid keypoints length: {len(keypoints)}, expected 194",
                    }
                    await manager.send_message(error_msg, session_id)
                    continue

                await manager.process_frame(session_id, keypoints)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
