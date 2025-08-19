# app/websockets/session_manager.py
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
        self.session_data: Dict[str, dict] = {}  # defaultdict 제거
        self.keypoint_buffers: Dict[str, deque] = {}  # defaultdict 제거
        self.gloss_collectors: Dict[str, GlossCollector] = {}  # defaultdict 제거

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """WebSocket 연결 등록"""
        # await websocket.accept() => ASGI protocol 위반
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "user_id": user_id,
            "start_time": time.time(),
            "frame_count": 0,
            "translation_active": False,  # 번역 상태 추가
        }

        logger.info(f"세션 연결: session_id={session_id}, user_id={user_id}")

    def start_translation(self, session_id: str) -> bool:
        """번역 세션 시작 - GlossCollector 생성"""
        if session_id not in self.session_data:
            logger.error(f"존재하지 않는 세션: {session_id}")
            return False

        # 새로운 GlossCollector 인스턴스 생성
        self.gloss_collectors[session_id] = GlossCollector(
            confidence_threshold=0.8,
            max_glosses=100,
            timeout_seconds=5,
            use_local_fallback=True,
        )

        # 키포인트 버퍼 초기화
        self.keypoint_buffers[session_id] = deque(maxlen=10)

        # 번역 상태 업데이트
        self.session_data[session_id]["translation_active"] = True
        self.session_data[session_id]["translation_start_time"] = time.time()

        logger.info(f"✅ 번역 시작: session_id={session_id}")
        logger.info(
            f"🔍 GlossCollector 생성: {self.gloss_collectors[session_id].get_status()}"
        )

        return True

    def stop_translation(self, session_id: str) -> Optional[GlossCollector]:
        """번역 세션 종료 - GlossCollector 반환 후 정리"""
        if session_id not in self.session_data:
            logger.warning(f"존재하지 않는 세션: {session_id}")
            return None

        # 번역 상태 업데이트
        self.session_data[session_id]["translation_active"] = False

        # GlossCollector 반환 (종료 시 문장 생성용)
        gloss_collector = self.gloss_collectors.get(session_id)

        if gloss_collector:
            logger.info(
                f"🔍 번역 종료 시 GlossCollector 상태: {gloss_collector.get_status()}"
            )

        # 번역 관련 리소스만 정리 (연결은 유지)
        if session_id in self.gloss_collectors:
            del self.gloss_collectors[session_id]
        if session_id in self.keypoint_buffers:
            del self.keypoint_buffers[session_id]

        logger.info(f"✅ 번역 종료: session_id={session_id}")
        return gloss_collector

    def disconnect(self, session_id: str):
        """WebSocket 연결 해제 - 모든 리소스 정리"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.session_data:
            del self.session_data[session_id]

        if session_id in self.keypoint_buffers:
            del self.keypoint_buffers[session_id]

        if session_id in self.gloss_collectors:
            del self.gloss_collectors[session_id]

        logger.info(f"세션 해제: session_id={session_id}")

    def is_translation_active(self, session_id: str) -> bool:
        """번역 활성 상태 확인"""
        session_data = self.session_data.get(session_id)
        return session_data and session_data.get("translation_active", False)

    def add_keypoints(self, session_id: str, keypoints: list) -> bool:
        """키포인트 데이터를 버퍼에 추가"""
        if session_id not in self.keypoint_buffers:
            logger.warning(f"키포인트 버퍼 없음: {session_id}")
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

    def get_gloss_collector(self, session_id: str) -> Optional[GlossCollector]:
        """세션별 Gloss 수집기 반환"""
        collector = self.gloss_collectors.get(session_id)
        if not collector:
            logger.warning(f"GlossCollector 없음: {session_id}")
        return collector

    def has_session(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        return session_id in self.session_data

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """세션 정보 반환"""
        return self.session_data.get(session_id)

    async def send_to_session(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"메시지 전송 실패: session_id={session_id}, error={e}")
                self.disconnect(session_id)

    def debug_sessions(self):
        """디버그: 현재 세션 상태 출력"""
        logger.info(f"🔍 활성 세션 수: {len(self.session_data)}")
        for session_id, data in self.session_data.items():
            collector = self.gloss_collectors.get(session_id)
            logger.info(f"🔍 세션 {session_id}:")
            logger.info(f"  - user_id: {data.get('user_id')}")
            logger.info(f"  - translation_active: {data.get('translation_active')}")
            logger.info(f"  - frame_count: {data.get('frame_count')}")
            logger.info(f"  - gloss_collector: {'있음' if collector else '없음'}")
            if collector and hasattr(collector, "get_status"):
                logger.info(f"  - collector_status: {collector.get_status()}")


# 싱글톤 인스턴스
session_manager = SessionManager()
