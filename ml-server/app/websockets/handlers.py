# app/websockets/handlers.py
import json
import logging
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, List

from .message_types import (
    MessageType,
    KeypointsMessage,
    StartTranslationMessage,
    StopTranslationMessage,
    PredictionResultMessage,
    SentenceGeneratedMessage,
    ErrorMessage,
)
from ..services.auth_service import auth_service
from ..models.model_manager import model_manager
from ..services.gloss_collector import gloss_collector
from .session_manager import session_manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self):
        # self.active_sessions: Dict[str, Dict] = {}
        pass  # use session_manager

    async def handle_connection(self, websocket: WebSocket, token: str):
        """WebSocket 연결 처리 - SessionManager 연동"""
        user_id = None
        session_id = None  # for 세션 추적
        try:
            logger.info(f"WebSocket 연결 시도: token={token[:20]}...")

            # 토큰 검증
            user_info = await auth_service.verify_token(token)

            if not user_info or not isinstance(user_info, dict):
                logger.warning(f"토큰 검증 실패: user_info={user_info}")

                # 대안으로 JWT 직접 디코딩
                user_id = auth_service.extract_user_id_from_token(token)
                if user_id:
                    user_info = {"user_id": user_id, "valid": True}
                    logger.info(f"JWT 직접 디코딩으로 user_id 추출: {user_id}")
                else:
                    await websocket.close(code=1008, reason="Invalid token")
                    return

            await websocket.accept()

            # user_id 추출
            if isinstance(user_info, dict):
                user_id = (
                    user_info.get("user_id")
                    or user_info.get("userId")
                    or user_info.get("sub")
                )

            if not user_id:
                logger.error(f"user_id를 찾을 수 없음: user_info={user_info}")
                await websocket.close(code=1008, reason="User ID not found")
                return

            user_id = str(user_id)
            logger.info(f"WebSocket 연결 승인: user_id={user_id}")

            # 메시지 루프
            while True:
                try:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)

                    # message로부터 session_id 추출하여 세션 등록
                    current_session_id = message_data.get("session_id")
                    if current_session_id and current_session_id != session_id:
                        # 새로운 세션 ID 발견 시 등록
                        if not session_manager.has_session(current_session_id):
                            await session_manager.connect(
                                websocket, current_session_id, user_id
                            )
                            session_id = current_session_id
                            logger.info(f"새로운 세션 등록: session_id={session_id}")

                    await self.handle_message(websocket, message_data, user_id)

                except WebSocketDisconnect:
                    logger.info(f"WebSocket 연결 종료: user_id={user_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                    await self.send_error(
                        websocket, "", "INVALID_JSON", "Invalid JSON format"
                    )
                except Exception as e:
                    logger.error(f"메시지 처리 오류: {e}")
                    await self.send_error(websocket, "", "MESSAGE_ERROR", str(e))

        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 해제: user_id={user_id}")
        except Exception as e:
            logger.error(f"WebSocket 연결 오류: {e}")
            logger.exception("상세 오류 정보:")
        finally:
            # 세션 정리 추가
            if session_id:
                session_manager.disconnect(session_id)
            await self.cleanup_user_sessions(user_id)

    async def cleanup_user_sessions(self, user_id: str):
        """사용자 세션 정리"""
        if not user_id:
            return

        sessions_to_remove = []
        for session_id, session_info in self.active_sessions.items():
            if session_info.get("user_id") == user_id:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            logger.info(f"세션 정리: session_id={session_id}, user_id={user_id}")

    async def handle_message(
        self, websocket: WebSocket, data: Dict[Any, Any], user_id: str
    ):
        """메시지 타입별 처리"""
        message_type = data.get("type")
        session_id = data.get("session_id")

        if not message_type:
            await self.send_error(
                websocket, "", "MISSING_TYPE", "Message type is required"
            )
            return

        if not session_id:
            await self.send_error(
                websocket, "", "MISSING_SESSION_ID", "Session ID is required"
            )
            return

        if not session_manager.has_session(session_id):
            logger.info(f"새 세션 등록: session_id={session_id}")
            session_manager.connect(websocket, session_id, user_id)
        try:
            if message_type == MessageType.START_TRANSLATION:
                await self.handle_start_translation(websocket, data, user_id)
            elif message_type == MessageType.STOP_TRANSLATION:
                await self.handle_stop_translation(websocket, data, user_id)
            elif message_type == MessageType.KEYPOINTS:
                await self.handle_keypoints(websocket, data, user_id)
            else:
                logger.warning(f"알 수 없는 메시지 타입: {message_type}")
                await self.send_error(
                    websocket,
                    session_id,
                    "UNKNOWN_MESSAGE_TYPE",
                    f"Unknown type: {message_type}",
                )

        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.exception("메시지 처리 오류 상세:")
            await self.send_error(websocket, session_id, "PROCESSING_ERROR", str(e))

    async def handle_register_session(
        self, websocket: WebSocket, data: Dict, user_id: str
    ):
        """세션 등록 처리"""
        try:
            from .message_types import RegisterSessionMessage

            message = RegisterSessionMessage(**data)
            session_id = message.session_id

            # 세션이 이미 존재하는지 확인
            if session_manager.has_session(session_id):
                logger.info(f"기존 세션 재사용: session_id={session_id}")
            else:
                # 새 세션 등록
                await session_manager.connect(websocket, session_id, user_id)
                logger.info(
                    f"✅ 새 세션 등록: session_id={session_id}, user_id={user_id}"
                )

            # 등록 완료 상태 전송
            await self.send_status(websocket, session_id, "session_registered")

        except Exception as e:
            logger.error(f"세션 등록 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "REGISTER_SESSION_ERROR", str(e)
            )

    async def handle_start_translation(
        self, websocket: WebSocket, data: Dict, user_id: str
    ):
        """번역 시작 처리 - SessionManager 사용"""
        try:
            message = StartTranslationMessage(**data)
            session_id = message.session_id

            logger.info(f"번역 시작 요청: session_id={session_id}")

            # ✅ SessionManager를 통한 번역 시작
            if session_manager.start_translation(session_id):
                logger.info(f"✅ 번역 세션 시작: session_id={session_id}")
                await self.send_status(websocket, session_id, "translation_started")
            else:
                logger.error(f"번역 시작 실패: session_id={session_id}")
                await self.send_error(
                    websocket,
                    session_id,
                    "START_TRANSLATION_ERROR",
                    "Failed to start translation",
                )

        except Exception as e:
            logger.error(f"번역 시작 처리 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "START_TRANSLATION_ERROR", str(e)
            )

    async def handle_stop_translation(
        self, websocket: WebSocket, data: Dict, user_id: str
    ):
        """번역 종료 처리 - SessionManager 사용"""
        try:
            message = StopTranslationMessage(**data)
            session_id = message.session_id

            logger.info(f"번역 종료 요청: session_id={session_id}")

            # SessionManager를 통한 번역 종료 및 GlossCollector 획득
            gloss_collector = session_manager.stop_translation(session_id)

            if (
                gloss_collector
                and hasattr(gloss_collector, "has_glosses")
                and gloss_collector.has_glosses()
            ):
                try:
                    current_glosses = gloss_collector.get_current_glosses()
                    logger.info(f"✅ 종료 시 수집된 gloss: {current_glosses}")

                    final_sentence = await gloss_collector.force_generate_sentence()

                    if final_sentence and final_sentence.strip():
                        await self.send_sentence(
                            websocket,
                            session_id,
                            final_sentence,
                            current_glosses,
                        )
                        logger.info(f"✅ 종료 시 문장 생성: {final_sentence}")
                    else:
                        logger.info("생성된 문장이 비어있음")

                except Exception as e:
                    logger.error(f"마지막 문장 생성 실패: {e}")
            else:
                logger.info(f"종료 시 수집된 gloss 없음: session_id={session_id}")

            await self.send_status(websocket, session_id, "translation_stopped")

        except Exception as e:
            logger.error(f"번역 종료 처리 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "STOP_TRANSLATION_ERROR", str(e)
            )

    async def handle_keypoints(self, websocket: WebSocket, data: Dict, user_id: str):
        """키포인트 처리 - SessionManager 사용"""
        try:
            message = KeypointsMessage(**data)
            session_id = message.session_id

            # ✅ 세션 및 번역 상태 확인
            if not session_manager.has_session(session_id):
                logger.warning(f"세션 없음: {session_id}")
                await self.send_error(
                    websocket, session_id, "SESSION_NOT_FOUND", "Session not found"
                )
                return

            if not session_manager.is_translation_active(session_id):
                logger.warning(f"번역 비활성 상태: {session_id}")
                await self.send_error(
                    websocket,
                    session_id,
                    "TRANSLATION_NOT_ACTIVE",
                    "Translation not active",
                )
                return

            # 키포인트 처리 (기존 로직)
            keypoints_array = np.array(message.keypoints, dtype=np.float32)

            if keypoints_array.shape != (10, 194):
                logger.error(f"잘못된 키포인트 형태: {keypoints_array.shape}")
                await self.send_error(
                    websocket,
                    session_id,
                    "INVALID_KEYPOINTS_SHAPE",
                    "Invalid keypoints shape",
                )
                return

            # 모델 예측
            keypoints_list = keypoints_array.tolist()
            prediction = await model_manager.predict_async(keypoints_list)

            if prediction is None:
                await self.send_error(
                    websocket,
                    session_id,
                    "PREDICTION_FAILED",
                    "Model prediction failed",
                )
                return

            label = prediction.get("label", "")
            confidence = float(prediction.get("confidence", 0.0))

            logger.info(f"예측 결과: label={label}, confidence={confidence:.3f}")

            # 예측 결과 전송
            await self.send_prediction(
                websocket, session_id, label, confidence, message.frame_index
            )

            # ✅ SessionManager를 통한 GlossCollector 획득
            gloss_collector = session_manager.get_gloss_collector(session_id)

            if gloss_collector:
                logger.info(f"🔍 GlossCollector 상태: {gloss_collector.get_status()}")

                sentence = await gloss_collector.add_prediction(label, confidence)

                if sentence:
                    logger.info(f"✅ 문장 생성: {sentence}")
                    await self.send_sentence(
                        websocket,
                        session_id,
                        sentence,
                        gloss_collector.get_current_glosses(),
                    )
            else:
                logger.error(f"❌ GlossCollector 없음: {session_id}")

        except Exception as e:
            logger.error(f"키포인트 처리 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "KEYPOINTS_ERROR", str(e)
            )

    async def cleanup_user_sessions(self, user_id: str):
        """사용자 세션 정리 - SessionManager 사용"""
        # SessionManager에서 해당 user_id의 모든 세션 정리
        sessions_to_remove = []
        for session_id, session_data in session_manager.session_data.items():
            if session_data.get("user_id") == user_id:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            session_manager.disconnect(session_id)
            logger.info(f"사용자 세션 정리: session_id={session_id}, user_id={user_id}")

    async def send_prediction(
        self,
        websocket: WebSocket,
        session_id: str,
        label: str,
        confidence: float,
        frame_index: int,
    ):
        """예측 결과 전송"""
        try:
            message = PredictionResultMessage(
                session_id=session_id,
                label=label,
                confidence=confidence,
                frame_index=frame_index,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.debug(f"예측 결과 전송: {label} ({confidence:.3f})")
        except Exception as e:
            logger.error(f"예측 결과 전송 실패: {e}")

    async def send_sentence(
        self, websocket: WebSocket, session_id: str, sentence: str, glosses: List[str]
    ):
        """문장 생성 결과 전송"""
        try:
            message = SentenceGeneratedMessage(
                session_id=session_id,
                sentence=sentence,
                glosses=glosses,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.info(f"문장 전송: {sentence}")
        except Exception as e:
            logger.error(f"문장 전송 실패: {e}")

    async def send_status(self, websocket: WebSocket, session_id: str, status: str):
        """상태 메시지 전송"""
        try:
            message = {
                "type": MessageType.STATUS,
                "session_id": session_id,
                "status": status,
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(message))
            logger.debug(f"상태 메시지 전송: {status}")
        except Exception as e:
            logger.error(f"상태 메시지 전송 실패: {e}")

    async def send_error(
        self, websocket: WebSocket, session_id: str, error_code: str, error_message: str
    ):
        """오류 메시지 전송"""
        try:
            message = ErrorMessage(
                session_id=session_id,
                error_code=error_code,
                error_message=error_message,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.warning(f"오류 메시지 전송: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"오류 메시지 전송 실패: {e}")


# 싱글톤 인스턴스
websocket_handler = WebSocketHandler()
