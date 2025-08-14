# session 및 buffer 관리
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.session: Dict[str, Dict] = {}
        self.session_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.MAX_BUFFER_SIZE)
        )
        self.timeout_minutes = settings.SESSION_TIMEOUT_MINUTES

    def create_session(
        self, session_id: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """새 세션 생성"""
        current_time = time.time()

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": current_time,
            "last_activity": current_time,
            "frame_count": 0,
            "total_predictions": 0,
            "status": "active",
        }

        self.sessions[session_id] = session_data
        logger.info(f"세션 생성: {session_id}")

        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # 세션 만료 확인
        if self._is_session_expired(session):
            self.cleanup_session(session_id)
            return None

        return session

    def update_session_activity(self, session_id: str):
        """세션 활동 시간 업데이트"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()

    def add_frame_result(
        self, session_id: str, frame_index: int, result: Dict[str, Any]
    ):
        """프레임 처리 결과를 세션 버퍼에 추가"""
        if session_id not in self.sessions:
            self.create_session(session_id)

        # 세션 정보 업데이트
        session = self.sessions[session_id]
        session["frame_count"] += 1
        session["total_predictions"] += 1
        self.update_session_activity(session_id)

        # 버퍼에 결과 추가
        buffer_item = {
            "frame_index": frame_index,
            "timestamp": time.time(),
            "result": result,
        }

        self.session_buffers[session_id].append(buffer_item)

        logger.debug(f"프레임 결과 저장: session={session_id}, frame={frame_index}")

    def get_recent_results(
        self, session_id: str, count: int = 10
    ) -> List[Dict[str, Any]]:
        """최근 처리 결과 조회"""
        if session_id not in self.session_buffers:
            return []

        buffer = self.session_buffers[session_id]
        return list(buffer)[-count:]

    def cleanup_session(self, session_id: str):
        """세션 정리"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if session_id in self.session_buffers:
            del self.session_buffers[session_id]

        logger.info(f"세션 정리: {session_id}")

    def cleanup_expired_sessions(self):
        """만료된 세션들 정리"""
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.cleanup_session(session_id)

        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 정리됨")

    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """세션 만료 여부 확인"""
        current_time = time.time()
        last_activity = session["last_activity"]
        timeout_seconds = self.timeout_minutes * 60

        return (current_time - last_activity) > timeout_seconds

    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계 정보"""
        active_sessions = len(self.sessions)
        total_frames = sum(session["frame_count"] for session in self.sessions.values())

        return {
            "active_sessions": active_sessions,
            "total_frames_processed": total_frames,
            "average_frames_per_session": total_frames / max(active_sessions, 1),
        }
