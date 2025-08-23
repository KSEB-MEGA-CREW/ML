# app/websockets/handlers.py
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
import numpy as np
import time

from app.websockets.session_manager import session_manager, SessionData
from app.websockets.message_types import (
    MessageType,
    MessageFactory,
    KeypointDataMessage,
    TranslationStartMessage,
    TranslationEndMessage,
    PingMessage,
)
from ..core.security import TokenVerifier
from app.models.model_manager import model_manager
from app.services.claude_service import claude_service
from ..services.gloss_collector import gloss_collector
from .session_manager import session_manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """WebSocket 연결 및 메시지 처리 (사용자 제어 번역)"""

    def __init__(self):
        self.session_manager = session_manager

    async def handle_connection(self, websocket: WebSocket, token: str):
        """WebSocket 연결 처리"""
        user_id = None
        session_id = None
        try:
            logger.info(f"WebSocket 연결 시도: token={token[:20]}...")

            # 1. 토큰 검증 (다중 단계)
            user_id = await self._verify_user_token(token)

            if not user_id:
                logger.error("모든 토큰 검증 방법 실패")
                await websocket.close(code=4001, reason="Authentication failed")
                return

            # 2. WebSocket 연결 승인
            await websocket.accept()
            logger.info(f"✅ WebSocket 연결 승인: 사용자 {user_id}")

            # 3. 세션 생성
            session_id = await self.session_manager.create_session(websocket, user_id)
            logger.info(f"✅ 세션 생성 완료: {session_id[:8]} (사용자: {user_id})")

            # 4. 연결 성공 메시지 전송
            status_msg = MessageFactory.create_translation_status(
                session_id=session_id,
                status="connected",
                message=f"세션 생성 완료: {session_id[:8]}",
                gloss_count=0,
            )
            await websocket.send_text(status_msg.model_dump_json())

            # 5. 메시지 루프 시작 (Keep-alive 포함)
            await self._message_loop_with_keepalive(websocket, session_id)

        except HTTPException as e:
            await websocket.close(
                code=4001, reason=f"Authentication failed: {e.detail}"
            )
            logger.warning(f"WebSocket 인증 실패: {e.detail}")

        except WebSocketDisconnect:
            logger.info(
                f"WebSocket 연결이 클라이언트에 의해 종료됨 (session: {session_id[:8] if session_id else 'unknown'})"
            )

        except Exception as e:
            logger.error(f"WebSocket 연결 처리 중 오류: {e}")
            import traceback

            logger.error(f"상세 오류: {traceback.format_exc()}")
            try:
                await websocket.close(code=1011, reason="Internal server error")
            except:
                pass

        finally:
            # 세션 정리
            if session_id:
                await self.session_manager.remove_session(session_id)
                logger.info(f"🧹 세션 정리 완료: {session_id[:8]}")
            else:
                await self.session_manager.remove_session_by_websocket(websocket)

    async def _verify_user_token(self, token: str) -> str:
        """다중 단계 토큰 검증"""
        try:
            # 🔥 0단계: 개발용 토큰 우선 처리 (JWT 디코딩 전에)
            if token in ['demo_token', 'test_token', 'dev_token']:
                logger.info(f"🚀 개발용 토큰 사용: {token}")
                return "demo_user"
            
            # 1단계: 백엔드 검증
            logger.info("🔍 1단계: 백엔드 토큰 검증 시도")
            user_id = await TokenVerifier.verify_token(token)

            if user_id:
                logger.info(f"✅ 백엔드 검증 성공: user_id={user_id}")
                return str(user_id)

            # 2단계: JWT 직접 디코딩
            logger.warning("⚠️ 백엔드 검증 실패 - JWT 직접 디코딩 시도")

            import jwt

            decoded = jwt.decode(token, options={"verify_signature": False})
            logger.info(f"🔍 JWT 페이로드: {decoded}")

            # 여러 필드에서 사용자 ID 추출 시도
            user_id = (
                decoded.get("userId") or decoded.get("user_id") or decoded.get("sub")
            )

            if user_id:
                logger.info(f"✅ JWT 직접 디코딩 성공: user_id={user_id}")
                return str(user_id)

            # 3단계: 이메일을 사용자 ID로 사용
            email = decoded.get("sub")  # subject는 보통 이메일
            if email and "@" in email:
                logger.warning(f"⚠️ 이메일을 사용자 ID로 사용: {email}")
                return email

            logger.error("❌ JWT에서 사용자 식별자를 찾을 수 없음")
            return None

        except Exception as e:
            logger.error(f"❌ 토큰 검증 오류: {e}")
            import traceback

            logger.error(f"상세 오류: {traceback.format_exc()}")
            return None

    async def _message_loop_with_keepalive(self, websocket: WebSocket, session_id: str):
        """Keep-alive 메커니즘이 포함된 메시지 처리 루프"""
        session_data = self.session_manager.get_session(session_id)
        if not session_data:
            await websocket.close(code=1011, reason="Session not found")
            return

        last_ping_time = time.time()
        ping_interval = 30  # 30초마다 핑 전송

        try:
            while True:
                try:
                    # 타임아웃을 짧게 설정하여 주기적으로 keep-alive 체크
                    message_text = await asyncio.wait_for(
                        websocket.receive_text(), timeout=1.0
                    )

                    try:
                        message_data = json.loads(message_text)
                        await self._handle_message(session_data, message_data)
                    except json.JSONDecodeError:
                        error_msg = MessageFactory.create_error(
                            session_id=session_id,
                            error_code="INVALID_JSON",
                            error_message="Invalid JSON format",
                        )
                        await websocket.send_text(error_msg.model_dump_json())
                    except Exception as e:
                        logger.error(
                            f"메시지 처리 중 오류 (세션 {session_id[:8]}): {e}"
                        )
                        error_msg = MessageFactory.create_error(
                            session_id=session_id,
                            error_code="MESSAGE_PROCESSING_ERROR",
                            error_message=str(e),
                        )
                        await websocket.send_text(error_msg.model_dump_json())

                except asyncio.TimeoutError:
                    # 타임아웃 발생 시 keep-alive 체크
                    current_time = time.time()
                    if current_time - last_ping_time > ping_interval:
                        try:
                            # 핑 메시지 전송
                            ping_msg = MessageFactory.create_ping_message(session_id)
                            await websocket.send_text(ping_msg.model_dump_json())
                            last_ping_time = current_time
                            logger.debug(f"📡 Keep-alive 핑 전송: {session_id[:8]}")
                        except Exception as ping_error:
                            logger.error(f"핑 전송 실패: {ping_error}")
                            break
                    continue

        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 종료 (세션 {session_id[:8]})")
        except Exception as e:
            logger.error(f"메시지 루프 오류 (세션 {session_id[:8]}): {e}")

    async def _handle_message(
        self, session_data: SessionData, message_data: Dict[str, Any]
    ):
        """개별 메시지 처리"""
        message_type = message_data.get("type")
        session_id = session_data.session_id

        # 세션 활동 시간 업데이트
        session_data.last_activity = time.time()

        if message_type == MessageType.KEYPOINTS:
            await self._handle_keypoint_data(session_data, message_data)

        elif message_type in [MessageType.TRANSLATION_START, "start_translation"]:
            await self._handle_translation_start(session_data, message_data)

        elif message_type in [MessageType.TRANSLATION_END, "stop_translation"]:
            await self._handle_translation_end(session_data, message_data)

        elif message_type == MessageType.PING:
            await self._handle_ping(session_data)

        elif message_type == "pong":
            logger.debug(f"Pong 수신: {session_id[:8]}")

        else:
            logger.warning(
                f"알 수 없는 메시지 타입: {message_type} (세션 {session_id[:8]})"
            )

    async def _handle_keypoint_data(
        self, session_data: SessionData, message_data: Dict[str, Any]
    ):
        """키포인트 데이터 처리"""
        try:
            keypoint_msg = KeypointDataMessage(**message_data)

            # 번역이 활성화된 경우에만 키포인트 처리
            if not session_data.gloss_collector.is_translation_active():
                logger.debug(
                    f"번역 비활성 상태 - 키포인트 무시: {session_data.session_id[:8]}"
                )
                return

            # 키포인트 버퍼에 추가
            is_batch_ready = session_data.add_keypoints(keypoint_msg.keypoints)

            if is_batch_ready:
                # 10프레임 배치가 준비됨 -> 모델 추론 실행
                await self._process_batch_prediction(session_data)

        except Exception as e:
            logger.error(f"키포인트 데이터 처리 오류: {e}")
            error_msg = MessageFactory.create_error(
                session_id=session_data.session_id,
                error_code="KEYPOINT_PROCESSING_ERROR",
                error_message=f"키포인트 처리 실패: {str(e)}",
            )
            await session_data.websocket.send_text(error_msg.model_dump_json())

    async def _process_batch_prediction(self, session_data: SessionData):
        """10프레임 배치 예측 처리"""
        try:
            # 배치 데이터 추출
            batch_keypoints = session_data.get_batch_keypoints()
            if not batch_keypoints:
                return

            # 모델 입력 형태로 변환: (10, 194) -> (1, 10, 194)
            batch_array = np.array(batch_keypoints).reshape(1, 10, 194)

            # 모델 추론 (비동기)
            prediction_result = await model_manager.predict_async(batch_array)

            if prediction_result:
                gloss = prediction_result["gloss"]
                confidence = prediction_result["confidence"]

                # 통계 업데이트
                session_data.update_stats(confidence)
                session_data.total_frames_processed += 10

                logger.debug(
                    f"예측 결과: {gloss} (신뢰도: {confidence:.3f}) - 세션 {session_data.session_id[:8]}"
                )

                # 예측 결과 전송 (번역 활성화 시에만)
                if session_data.gloss_collector.is_translation_active():
                    result_msg = MessageFactory.create_prediction_result(
                        session_id=session_data.session_id,
                        gloss=gloss,
                        confidence=confidence,
                        frame_count=10,
                    )
                    await session_data.websocket.send_text(result_msg.model_dump_json())

                # 고신뢰도 gloss 수집 (번역 활성화 시에만)
                gloss_added = session_data.gloss_collector.add_gloss(gloss, confidence)

                if gloss_added:
                    # 현재 gloss 수집 상태 전송
                    await self._send_gloss_status_update(session_data)

        except Exception as e:
            logger.error(f"배치 예측 처리 오류: {e}")
            error_msg = MessageFactory.create_error(
                session_id=session_data.session_id,
                error_code="PREDICTION_ERROR",
                error_message=f"예측 처리 실패: {str(e)}",
            )
            await session_data.websocket.send_text(error_msg.model_dump_json())

    async def _handle_translation_start(
        self, session_data: SessionData, message_data: Dict[str, Any]
    ):
        """번역 시작 처리"""
        try:
            translation_msg = TranslationStartMessage(**message_data)

            # 번역 시작
            session_data.gloss_collector.start_translation()

            # 세션 상태 초기화
            session_data.frame_buffer.clear()
            session_data.frame_count = 0
            session_data.is_active = True

            # 시작 확인 메시지 전송
            status_msg = MessageFactory.create_translation_status(
                session_id=session_data.session_id,
                status="translation_started",
                message="번역이 시작되었습니다. 수어를 시작해 주세요.",
                gloss_count=0,
            )
            await session_data.websocket.send_text(status_msg.model_dump_json())

            logger.info(
                f"✅ 번역 시작: {session_data.session_id[:8]} (사용자: {session_data.user_id})"
            )

        except Exception as e:
            logger.error(f"번역 시작 처리 오류: {e}")

    async def _handle_translation_end(
        self, session_data: SessionData, message_data: Dict[str, Any]
    ):
        """번역 종료 처리"""
        try:
            translation_msg = TranslationEndMessage(**message_data)

            # 번역 종료 및 번역 수행 여부 확인
            should_translate = session_data.gloss_collector.stop_translation()

            if should_translate:
                # 수집된 gloss로 번역 수행
                await self._generate_translation(session_data, "user_end")
            else:
                # 수집된 gloss가 없음
                status_msg = MessageFactory.create_translation_status(
                    session_id=session_data.session_id,
                    status="translation_ended_no_result",
                    message="번역할 수어 단어가 수집되지 않았습니다.",
                    gloss_count=0,
                )
                await session_data.websocket.send_text(status_msg.model_dump_json())

            # 세션 상태 업데이트
            session_data.is_active = False

            logger.info(
                f"🔴 번역 종료: {session_data.session_id[:8]} (번역 수행: {should_translate})"
            )

        except Exception as e:
            logger.error(f"번역 종료 처리 오류: {e}")

    async def _generate_translation(
        self, session_data: SessionData, trigger: str = "user_end"
    ):
        """Claude API를 사용한 문장 생성"""
        try:
            # 수집된 gloss 가져오기
            gloss_sequence = session_data.gloss_collector.get_collected_glosses()

            if not gloss_sequence:
                return

            logger.info(
                f"🔄 번역 생성 시작: {' → '.join(gloss_sequence)} (세션 {session_data.session_id[:8]})"
            )

            # Claude API로 문장 생성
            translation_result = await claude_service.generate_sentence(gloss_sequence)

            if translation_result:
                # 번역 결과 전송
                translation_msg = MessageFactory.create_translation_result(
                    session_id=session_data.session_id,
                    sentence=translation_result["sentence"],
                    gloss_sequence=gloss_sequence,
                    confidence_avg=session_data.gloss_collector.get_average_confidence(),
                    translation_trigger=trigger,
                )
                await session_data.websocket.send_text(
                    translation_msg.model_dump_json()
                )

                # gloss 수집기 초기화
                session_data.gloss_collector.reset_after_translation()

                logger.info(
                    f"✅ 번역 완료 (세션 {session_data.session_id[:8]}): {translation_result['sentence']}"
                )
            else:
                # 번역 실패
                error_msg = MessageFactory.create_error(
                    session_id=session_data.session_id,
                    error_code="TRANSLATION_FAILED",
                    error_message="Claude API 번역에 실패했습니다.",
                )
                await session_data.websocket.send_text(error_msg.model_dump_json())

        except Exception as e:
            logger.error(f"번역 생성 오류: {e}")
            error_msg = MessageFactory.create_error(
                session_id=session_data.session_id,
                error_code="TRANSLATION_ERROR",
                error_message=f"번역 생성 실패: {str(e)}",
            )
            await session_data.websocket.send_text(error_msg.model_dump_json())

    async def _send_gloss_status_update(self, session_data: SessionData):
        """현재 gloss 수집 상태 업데이트 전송"""
        try:
            gloss_count = session_data.gloss_collector.get_gloss_count()
            gloss_list = session_data.gloss_collector.get_collected_glosses()

            status_msg = MessageFactory.create_translation_status(
                session_id=session_data.session_id,
                status="gloss_collected",
                message=(
                    f"수어 단어 수집됨: {' → '.join(gloss_list[-3:])}..."
                    if len(gloss_list) > 3
                    else f"수어 단어: {' → '.join(gloss_list)}"
                ),
                gloss_count=gloss_count,
            )
            await session_data.websocket.send_text(status_msg.model_dump_json())

            logger.debug(
                f"📊 Gloss 상태 업데이트: {gloss_count}개 수집 (세션 {session_data.session_id[:8]})"
            )

        except Exception as e:
            logger.error(f"gloss 상태 업데이트 전송 오류: {e}")

    async def _handle_ping(self, session_data: SessionData):
        """핑 메시지 처리"""
        try:
            pong_msg = MessageFactory.create_pong_message(session_data.session_id)
            await session_data.websocket.send_text(pong_msg.model_dump_json())
            logger.debug(f"🏓 Pong 응답: {session_data.session_id[:8]}")
        except Exception as e:
            logger.error(f"핑 처리 오류: {e}")

    async def cleanup_all_sessions(self):
        """모든 세션 정리 (서버 종료 시)"""
        logger.info("🧹 모든 세션 정리 시작")
        await self.session_manager.cleanup_all_sessions()
        logger.info("✅ 모든 세션 정리 완료")


# 전역 WebSocket 핸들러 인스턴스
websocket_handler = WebSocketHandler()
