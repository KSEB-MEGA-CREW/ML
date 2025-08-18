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

    async def process_keypoints_array(
        self, session_id: str, keypoints_array: List[List[float]]
    ) -> bool:
        """프론트엔드에서 오는 키포인트 배열 처리"""
        if session_id not in self.session_buffers:
            return False

        try:
            # 각 프레임의 키포인트 개수 확인
            logger.debug(f"Received {len(keypoints_array)} frames")
            for i, frame in enumerate(keypoints_array):
                logger.debug(f"Frame {i}: {len(frame)} keypoints")

            # 키포인트 배열을 버퍼에 추가
            for frame_keypoints in keypoints_array:
                # 프레임별 키포인트가 194개가 아닐 수 있으므로 패딩 또는 조정
                if len(frame_keypoints) < 194:
                    # 부족한 경우 0으로 패딩
                    padded_keypoints = frame_keypoints + [0.0] * (
                        194 - len(frame_keypoints)
                    )
                elif len(frame_keypoints) > 194:
                    # 초과한 경우 194개만 사용
                    padded_keypoints = frame_keypoints[:194]
                else:
                    padded_keypoints = frame_keypoints

                self.session_buffers[session_id].append(padded_keypoints)

            # 10프레임이 누적되면 예측 수행
            if len(self.session_buffers[session_id]) >= 10:
                # 최근 10프레임 사용
                recent_frames = list(self.session_buffers[session_id])[-10:]
                logger.info(f"Processing prediction with {len(recent_frames)} frames")

                result = await self.predictor.predict_sequence(recent_frames)

                await self.send_message(result, session_id)

                # 버퍼의 절반 클리어 (슬라이딩 윈도우)
                for _ in range(5):
                    if len(self.session_buffers[session_id]) > 0:
                        self.session_buffers[session_id].popleft()

                return True

        except Exception as e:
            logger.error(f"Keypoints processing error for session {session_id}: {e}")
            error_message = {
                "success": False,
                "error": f"Processing failed: {str(e)}",
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
                keypoints_array = message["keypoints"]

                # 데이터 구조 확인 로그
                print(f"=== 데이터 구조 분석 ===")
                print(f"keypoints 타입: {type(keypoints_array)}")
                print(f"keypoints 길이: {len(keypoints_array)}")
                print(f"첫 번째 요소 타입: {type(keypoints_array[0])}")
                print(f"첫 번째 요소 길이: {len(keypoints_array[0])}")
                print(f"첫 번째 요소 샘플: {keypoints_array[0][:5]}...")
                print(f"========================")

                # 키포인트 배열 정보 로깅
                logger.debug(
                    f"Received keypoints array with {len(keypoints_array)} frames"
                )

                # 키포인트 배열 처리
                await manager.process_keypoints_array(session_id, keypoints_array)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
