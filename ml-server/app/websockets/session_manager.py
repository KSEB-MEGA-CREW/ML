# app/websockets/session_manager.py
import asyncio
import uuid
from collections import deque
from typing import Dict, Optional, Set
from fastapi import WebSocket
import time
import logging

from app.services.gloss_collector import GlossCollector
from app.websockets.message_types import MessageFactory, TranslationStatusMessage

logger = logging.getLogger(__name__)


# app/websockets/session_manager.py (SessionData 클래스 부분만 수정)


class SessionData:
    """개별 세션 데이터 클래스 (사용자 제어 번역)"""

    def __init__(self, session_id: str, user_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.user_id = user_id
        self.websocket = websocket
        self.created_at = time.time()
        self.last_activity = time.time()

        # 수어 인식 관련
        self.frame_buffer = deque(maxlen=10)  # 10프레임 버퍼
        self.gloss_collector = GlossCollector()
        self.frame_count = 0
        self.is_active = False  # 기본값을 False로 변경 (번역 시작 시에만 True)

        # 성능 모니터링
        self.total_frames_processed = 0
        self.total_predictions = 0
        self.avg_confidence = 0.0

        # 번역 세션 통계
        self.translation_sessions = 0
        self.total_glosses_collected = 0

    def add_keypoints(self, keypoints: list) -> bool:
        """키포인트를 버퍼에 추가하고 10프레임이 모였는지 확인"""
        # 번역이 활성화된 경우에만 키포인트 추가
        if not self.gloss_collector.is_translation_active():
            return False

        self.frame_buffer.append(keypoints)
        self.frame_count += 1
        self.last_activity = time.time()

        return len(self.frame_buffer) == 10

    def get_batch_keypoints(self) -> list:
        """10프레임 배치 데이터를 반환하고 버퍼 초기화"""
        if len(self.frame_buffer) != 10:
            return None

        batch = list(self.frame_buffer)
        self.frame_buffer.clear()
        return batch

    def start_new_translation_session(self):
        """새 번역 세션 시작"""
        self.translation_sessions += 1
        self.frame_buffer.clear()
        self.frame_count = 0
        self.is_active = True
        self.gloss_collector.start_translation()

    def end_translation_session(self):
        """번역 세션 종료"""
        self.is_active = False
        collected_count = self.gloss_collector.get_gloss_count()
        self.total_glosses_collected += collected_count

        return collected_count > 0

    def update_stats(self, confidence: float):
        """통계 업데이트"""
        self.total_predictions += 1
        self.avg_confidence = (
            self.avg_confidence * (self.total_predictions - 1) + confidence
        ) / self.total_predictions

    def get_translation_stats(self) -> dict:
        """번역 관련 통계 반환"""
        return {
            "translation_sessions": self.translation_sessions,
            "total_glosses_collected": self.total_glosses_collected,
            "current_gloss_count": self.gloss_collector.get_gloss_count(),
            "translation_active": self.gloss_collector.is_translation_active(),
            "translation_duration": self.gloss_collector.get_translation_duration(),
        }

    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """세션 만료 확인 (기본 5분)"""
        return time.time() - self.last_activity > timeout_seconds


class WebSocketSessionManager:
    """WebSocket 세션 관리자"""

    def __init__(self):
        # 세션 저장소 (메모리 기반)
        self.active_sessions: Dict[str, SessionData] = {}
        self.user_connections: Dict[str, WebSocket] = {}
        self.websocket_to_session: Dict[WebSocket, str] = {}

        # 정리 작업용
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """백그라운드 정리 작업 시작"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """주기적으로 만료된 세션 정리"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 정리
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"세션 정리 중 오류: {e}")

    async def create_session(self, websocket: WebSocket, user_id: str) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())

        # 기존 사용자 세션이 있다면 정리
        if user_id in self.user_connections:
            await self.remove_session_by_user(user_id)

        # 새 세션 생성
        session_data = SessionData(session_id, user_id, websocket)

        # 저장소에 등록
        self.active_sessions[session_id] = session_data
        self.user_connections[user_id] = websocket
        self.websocket_to_session[websocket] = session_id

        logger.info(
            f"새 세션 생성: {session_id[:8]} (사용자: {user_id}) "
            f"[총 {len(self.active_sessions)}개 세션]"
        )

        return session_id

    async def remove_session(self, session_id: str) -> bool:
        """세션 제거"""
        if session_id not in self.active_sessions:
            return False

        session_data = self.active_sessions[session_id]

        # 저장소에서 제거
        del self.active_sessions[session_id]

        if session_data.user_id in self.user_connections:
            del self.user_connections[session_data.user_id]

        if session_data.websocket in self.websocket_to_session:
            del self.websocket_to_session[session_data.websocket]

        logger.info(
            f"세션 제거: {session_id[:8]} (사용자: {session_data.user_id}) "
            f"[총 {len(self.active_sessions)}개 세션]"
        )

        return True

    async def remove_session_by_websocket(self, websocket: WebSocket) -> bool:
        """WebSocket 연결로 세션 제거"""
        if websocket not in self.websocket_to_session:
            return False

        session_id = self.websocket_to_session[websocket]
        return await self.remove_session(session_id)

    async def remove_session_by_user(self, user_id: str) -> bool:
        """사용자 ID로 세션 제거"""
        for session_id, session_data in self.active_sessions.items():
            if session_data.user_id == user_id:
                return await self.remove_session(session_id)
        return False

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """세션 데이터 조회"""
        return self.active_sessions.get(session_id)

    def get_session_by_websocket(self, websocket: WebSocket) -> Optional[SessionData]:
        """WebSocket으로 세션 조회"""
        if websocket not in self.websocket_to_session:
            return None

        session_id = self.websocket_to_session[websocket]
        return self.get_session(session_id)

    async def cleanup_expired_sessions(self):
        """만료된 세션들 정리"""
        expired_sessions = []

        for session_id, session_data in self.active_sessions.items():
            if session_data.is_expired():
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.remove_session(session_id)
            logger.info(f"만료된 세션 정리: {session_id[:8]}")

    async def cleanup_all_sessions(self):
        """모든 세션 정리 (서버 종료 시)"""
        session_ids = list(self.active_sessions.keys())

        for session_id in session_ids:
            session_data = self.active_sessions[session_id]
            try:
                # 클라이언트에게 종료 알림
                status_msg = MessageFactory.create_translation_status(
                    session_id=session_id,
                    status="server_shutdown",
                    message="서버가 종료됩니다",
                )
                await session_data.websocket.send_text(status_msg.model_dump_json())
            except:
                pass

            await self.remove_session(session_id)

        # 정리 작업 취소
        if self._cleanup_task:
            self._cleanup_task.cancel()

    def get_stats(self) -> dict:
        """세션 통계 정보"""
        total_frames = sum(
            s.total_frames_processed for s in self.active_sessions.values()
        )
        total_predictions = sum(
            s.total_predictions for s in self.active_sessions.values()
        )

        return {
            "active_sessions": len(self.active_sessions),
            "total_frames_processed": total_frames,
            "total_predictions": total_predictions,
            "avg_session_confidence": (
                sum(s.avg_confidence for s in self.active_sessions.values())
                / len(self.active_sessions)
                if self.active_sessions
                else 0
            ),
        }


# 전역 세션 매니저 인스턴스
session_manager = WebSocketSessionManager()
