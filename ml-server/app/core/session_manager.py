import time
import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.MAX_BUFFER_SIZE)
        )
        self.timeout_minutes = settings.SESSION_TIMEOUT_MINUTES
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def create_session(
        self, session_id: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create new session"""
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
        logger.info(f"Session created: {session_id}")

        return session_data

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check if session is expired
        if await self._is_session_expired(session):
            await self.cleanup_session(session_id)
            return None

        return session

    async def update_session_activity(self, session_id: str):
        """Update session activity time"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()

    async def add_frame_result(
        self, session_id: str, frame_index: int, result: Dict[str, Any]
    ):
        """Add frame processing result to session buffer"""
        if session_id not in self.sessions:
            await self.create_session(session_id)

        # Update session info
        session = self.sessions[session_id]
        session["frame_count"] += 1
        session["total_predictions"] += 1
        await self.update_session_activity(session_id)

        # Add result to buffer
        buffer_item = {
            "frame_index": frame_index,
            "timestamp": time.time(),
            "result": result,
        }

        self.session_buffers[session_id].append(buffer_item)

        logger.debug(f"Frame result stored: session={session_id}, frame={frame_index}")

    async def get_recent_results(
        self, session_id: str, count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent processing results"""
        if session_id not in self.session_buffers:
            return []

        buffer = self.session_buffers[session_id]
        return list(buffer)[-count:]

    async def cleanup_session(self, session_id: str):
        """Clean up session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if session_id in self.session_buffers:
            del self.session_buffers[session_id]

        logger.info(f"Session cleaned up: {session_id}")

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if await self._is_session_expired(session):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.cleanup_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Check if session is expired"""
        current_time = time.time()
        last_activity = session["last_activity"]
        timeout_seconds = self.timeout_minutes * 60

        return (current_time - last_activity) > timeout_seconds

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        active_sessions = len(self.sessions)
        total_frames = sum(session["frame_count"] for session in self.sessions.values())

        return {
            "active_sessions": active_sessions,
            "total_frames_processed": total_frames,
            "average_frames_per_session": total_frames / max(active_sessions, 1),
        }

    def __del__(self):
        """Clean up resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
