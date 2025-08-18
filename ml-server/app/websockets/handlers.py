# process Websocket endpoints and messages
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
import json
import time
import logging
from typing import Dict, Any

from .session_manager import session_manager
from .message_types import *
from app.models.model_manager import model_manager
from app.core.dependencies import verify_token_with_backend

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self):
        """WebSocket 핸들러 초기화"""
        self.session_manager = session_manager
        self.model_manager = model_manager

    async def handle_connection(self, websocket: WebSocket, token: str):
        """WebSocket 연결 처리"""
        try:
            # 1. 토큰 검증
            user_id = await verify_token_with_backend(token)
            if not user_id:
                await websocket.close(code=4001, reason="Invalid token")
                return

            # 2. 세션 ID 생성
            session_id = f"{user_id}_{int(time.time()*1000)}"

            # 3. 세션 등록
            await self.session_manager.connect(websocket, session_id, user_id)

            # 4. 연결 확인 메시지 전송
            await self._send_status_message(
                session_id,
                "connected",
                {
                    "user_id": user_id,
                    "model_ready": self.model_manager.is_model_ready(),
                },
            )

            # 5. 메시지 루프 시작
            await self._message_loop(websocket, session_id)

        except HTTPException as e:
            logger.warning(f"인증 실패: {e.detail}")
            await websocket.close(code=4001, reason=e.detail)

        except WebSocketDisconnect:
            logger.info(f"WebSocket 정상 종료: session_id={session_id}")

        except Exception as e:
            logger.error(f"WebSocket 연결 처리 실패: {e}")
            await websocket.close(code=4000, reason="Internal server error")

        finally:
            if "session_id" in locals():
                self.session_manager.disconnect(session_id)

    async def _message_loop(self, websocket: WebSocket, session_id: str):
        """메시지 처리 루프"""
        while True:
            try:
                # 메시지 수신
                message_data = await websocket.receive_text()
                message_dict = json.loads(message_data)

                # 메시지 타입별 처리
                await self._process_message(session_id, message_dict)

            except WebSocketDisconnect:
                break

            except json.JSONDecodeError as e:
                await self._send_error_message(session_id, "INVALID_JSON", str(e))

            except Exception as e:
                logger.error(f"메시지 처리 실패: session_id={session_id}, error={e}")
                await self._send_error_message(session_id, "PROCESSING_ERROR", str(e))

    async def _process_message(self, session_id: str, message_dict: Dict[str, Any]):
        """메시지 타입별 처리"""
        message_type = message_dict.get("type")

        if message_type == MessageType.KEYPOINTS:
            await self._handle_keypoints_message(session_id, message_dict)
        else:
            await self._send_error_message(
                session_id, "UNKNOWN_MESSAGE_TYPE", f"Unknown type: {message_type}"
            )

    async def _handle_keypoints_message(
        self, session_id: str, message_dict: Dict[str, Any]
    ):
        """키포인트 메시지 처리"""
        try:
            # 메시지 파싱
            message = KeypointsMessage(**message_dict)

            # 키포인트 버퍼에 추가
            is_batch_ready = self.session_manager.add_keypoints(
                session_id, message.keypoints
            )

            if is_batch_ready:
                # 10프레임 배치가 준비되면 예측 수행
                await self._perform_prediction(session_id, message.frame_index)

        except Exception as e:
            logger.error(f"키포인트 메시지 처리 실패: {e}")
            await self._send_error_message(session_id, "KEYPOINTS_ERROR", str(e))

    async def _perform_prediction(self, session_id: str, frame_index: int):
        """수어 예측 수행"""
        try:
            # 배치 키포인트 가져오기
            batch_keypoints = self.session_manager.get_batch_keypoints(session_id)
            if not batch_keypoints:
                return

            # 모델 예측
            label, confidence = await self.model_manager.predict_sign_language(
                batch_keypoints
            )

            # 예측 결과 전송
            await self._send_prediction_result(
                session_id, label, confidence, frame_index
            )

            # Gloss 수집기에 추가
            gloss_collector = self.session_manager.get_gloss_collector(session_id)
            sentence = await gloss_collector.add_prediction(label, confidence)

            # 문장이 생성되었으면 전송
            if sentence:
                await self._send_sentence_generated(
                    session_id, sentence, list(gloss_collector.glosses)
                )

        except Exception as e:
            logger.error(f"예측 수행 실패: session_id={session_id}, error={e}")
            await self._send_error_message(session_id, "PREDICTION_ERROR", str(e))

    async def _send_prediction_result(
        self, session_id: str, label: str, confidence: float, frame_index: int
    ):
        """예측 결과 전송"""
        message = PredictionResultMessage(
            session_id=session_id,
            label=label,
            confidence=confidence,
            frame_index=frame_index,
            timestamp=time.time(),
        )
        await self.session_manager.send_to_session(session_id, message.dict())

    async def _send_sentence_generated(
        self, session_id: str, sentence: str, glosses: List[str]
    ):
        """문장 생성 결과 전송"""
        message = SentenceGeneratedMessage(
            session_id=session_id,
            sentence=sentence,
            glosses=glosses,
            timestamp=time.time(),
        )
        await self.session_manager.send_to_session(session_id, message.dict())

    async def _send_error_message(
        self, session_id: str, error_code: str, error_message: str
    ):
        """에러 메시지 전송"""
        message = ErrorMessage(
            session_id=session_id,
            error_code=error_code,
            error_message=error_message,
            timestamp=time.time(),
        )
        await self.session_manager.send_to_session(session_id, message.dict())

    async def _send_status_message(
        self, session_id: str, status: str, details: dict = None
    ):
        """상태 메시지 전송"""
        message = StatusMessage(
            session_id=session_id, status=status, details=details, timestamp=time.time()
        )
        await self.session_manager.send_to_session(session_id, message.dict())


# 핸들러 인스턴스
websocket_handler = WebSocketHandler()
