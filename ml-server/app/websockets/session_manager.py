# manage sesssion state
import asyncio
import logging
from collections import deque, defaultdict
from typing import Dict, Optional, Deque
import time
import uuid

logger = logging.getLogger(__name__)


class SessionManager:  # session manager
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.sessions: Dict[str, Dict] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id mapping
        self.frame_buffers: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=10)
        )  # process per 10 frames
        self._initialized = True

        logger.info("SessionManager initialized")

    def create_session(self, user_id: str, websocket) -> str:
        """Create new session for user"""
        session_id = str(uuid.uuid4())

        # Clean up old session if exists
        if user_id in self.user_sessions:
            old_session_id = self.user_sessions[user_id]
            self.cleanup_session(old_session_id)

        # Create new session
        self.sessions[session_id] = {
            "user_id": user_id,  # mapping user_id
            "websocket": websocket,
            "created_at": time.time(),
            "frame_count": 0,  # initialize frame count
            "prediction_count": 0,
        }

        self.user_sessions[user_id] = session_id
        self.frame_buffers[session_id] = deque(maxlen=10)

        logger.info(f"Session created: {session_id} for user: {user_id}")
        return session_id

    def add_frame(self, session_id: str, keypoints: list) -> bool:
        """add frame to session buffer, if buffer is full => return true"""
        if session_id not in self.sessions:
            return False

        buffer = self.frame_buffers[session_id]
        buffer.append(keypoints)

        # update frame count
        self.sessions[session_id]["frame_count"] += 1

        return len(buffer) == 10

    def get_frame_buffer(self, session_id: str) -> Optional[Deque]:
        """get frame buffer for session"""
        return self.frame_buffers.get(session_id)

    def cleanup_session(self, session_id: str):
        """clean up session data"""
        if session_id in self.sessions:
            user_id = self.sessions[session_id]["user_id"]

            # remove from mappings
            del self.sessions[session_id]
            # delete by user_id or session_id
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            if session_id in self.frame_buffers:
                del self.frame_buffers[session_id]

            logger.info(f"Session cleaned up: {session_id}")

    def get_active_sessions_count(self) -> int:
        """get number of active sessions for monitoring"""
        return len(self.sessions)

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """get session statistics"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                "session_id": session_id,
                "user_id": session["user_id"],
                "frame_count": session["frame_count"],
                "prediction_count": session["prediction_count"],
                "duration": time.time() - session["created_at"],
            }
        return None
