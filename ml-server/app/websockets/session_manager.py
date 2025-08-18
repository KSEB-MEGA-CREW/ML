# manage sesssion state
from collections import defaultdict, deque
from typing import Dict, Optional
import time
import logging
from fastapi import WebSocket
from app.services.gloss_collector import GlossCollector

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        """WebSocket 세션 관리자 초기화"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, dict] = defaultdict(dict)
        self.keypoint_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.gloss_collectors: Dict[str, GlossCollector] = defaultdict(GlossCollector)

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """WebSocket 연결 등록"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "user_id": user_id,
            "start_time": time.time(),
            "frame_count": 0,
        }

        logger.info(f"세션 연결: session_id={session_id}, user_id={user_id}")

    def disconnect(self, session_id: str):
        """WebSocket 연결 해제"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.session_data:
            del self.session_data[session_id]

        if session_id in self.keypoint_buffers:
            del self.keypoint_buffers[session_id]

        if session_id in self.gloss_collectors:
            del self.gloss_collectors[session_id]

        logger.info(f"세션 해제: session_id={session_id}")

    def add_keypoints(self, session_id: str, keypoints: list) -> bool:
        """키포인트 데이터를 버퍼에 추가"""
        if session_id not in self.keypoint_buffers:
            return False

        self.keypoint_buffers[session_id].append(keypoints)
        self.session_data[session_id]["frame_count"] += 1

        # 10프레임이 모이면 True 반환
        return len(self.keypoint_buffers[session_id]) == 10

    def get_batch_keypoints(self, session_id: str) -> Optional[list]:
        """10프레임 배치 키포인트 반환"""
        if session_id not in self.keypoint_buffers:
            return None

        buffer = self.keypoint_buffers[session_id]
        if len(buffer) == 10:
            batch = list(buffer)
            buffer.clear()  # 버퍼 초기화
            return batch

        return None

    def get_gloss_collector(self, session_id: str) -> GlossCollector:
        """세션별 Gloss 수집기 반환"""
        return self.gloss_collectors[session_id]

    async def send_to_session(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"메시지 전송 실패: session_id={session_id}, error={e}")
                self.disconnect(session_id)


# 싱글톤 인스턴스
session_manager = SessionManager()
