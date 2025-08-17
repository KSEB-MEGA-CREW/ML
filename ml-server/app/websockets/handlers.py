import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.core.session_manager import SessionManager
from app.core.security import TokenVerifier
from app.models.predictor import SignLanguagePredictor
from app.websockets.connection_manager import ConnectionManager
from app.utils.keypoint_processor import KeypointProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """WebSocket endpoint for real-time sign language recognition"""

    session_id = None

    try:
        # Verify token
        user_id = await TokenVerifier.verify_token(token)
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token")
            return

        # Initialize managers (singletons)
        session_manager = SessionManager()
        connection_manager = ConnectionManager()
        predictor = SignLanguagePredictor()
        keypoint_processor = KeypointProcessor()

        # Create session
        session_id = session_manager.create_session(user_id, websocket)

        # Accept connection
        await connection_manager.connect(websocket, session_id)

        # Send connection confirmation
        await connection_manager.send_message(
            session_id,
            {
                "type": "connection_established",
                "session_id": session_id,
                "user_id": user_id,
                "message": "WebSocket 연결이 성공적으로 설정되었습니다.",
            },
        )

        logger.info(f"🔌 User {user_id} connected with session {session_id}")

        while True:
            # Receive message
            data = await websocket.receive_json()

            # Validate message format
            if not all(key in data for key in ["keypoints", "frame_index"]):
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "error",
                        "message": "Invalid message format. Required: keypoints, frame_index",
                    },
                )
                continue

            # Process keypoints
            keypoints = data["keypoints"]

            # Validate keypoints
            if not keypoint_processor.validate_keypoints(keypoints):
                await connection_manager.send_message(
                    session_id, {"type": "error", "message": "Invalid keypoints format"}
                )
                continue

            # Normalize keypoints
            normalized_keypoints = keypoint_processor.normalize_keypoints(keypoints)

            # Add frame to buffer (fixed method name)
            is_buffer_full = session_manager.add_frame(session_id, normalized_keypoints)

            if is_buffer_full:
                # Get frame buffer
                buffer = session_manager.get_frame_buffer(session_id)
                keypoints_sequence = list(buffer)

                # Make prediction
                prediction_result = await predictor.predict_sequence(keypoints_sequence)

                if prediction_result["success"]:
                    # Send prediction result
                    response = {
                        "type": "prediction",
                        "session_id": session_id,
                        "frame_index": data["frame_index"],
                        "result": prediction_result["prediction"],
                        "timestamp": prediction_result["timestamp"],
                    }

                    await connection_manager.send_message(session_id, response)

                    # Update prediction count
                    session_manager.sessions[session_id]["prediction_count"] += 1

                    logger.info(
                        f"🎯 Prediction sent to {session_id}: {prediction_result['prediction']['label']}"
                    )
                else:
                    # Send error
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "error",
                            "message": f"Prediction failed: {prediction_result['error']}",
                        },
                    )

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup
        if session_id:
            session_manager.cleanup_session(session_id)
            connection_manager.disconnect(session_id)
            logger.info(f"🧹 Session {session_id} cleaned up")
