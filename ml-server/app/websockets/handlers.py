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

    # Verify token
    user_id = await TokenVerifier.verify_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    # Initialize managers
    session_manager = SessionManager()
    connection_manager = ConnectionManager()
    predictor = SignLanguagePredictor()
    keypoint_processor = KeypointProcessor()

    # Create session
    session_id = session_manager.create_session(user_id, websocket)

    try:
        # Accept connection
        await connection_manager.connect(websocket, session_id)

        # Send connection confirmation
        await connection_manager.send_message(
            session_id,
            {
                "type": "connection_established",
                "session_id": session_id,
                "user_id": user_id,
            },
        )

        while True:
            # Receive message
            data = await websocket.receive_json()

            # Validate message format
            if not all(key in data for key in ["keypoints", "frame_index"]):
                await connection_manager.send_message(
                    session_id, {"type": "error", "message": "Invalid message format"}
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

            # Add frame to buffer
            is_buffer_full = session_manager.add_frame(session_id, keypoints)

            if is_buffer_full:
                # Get frame buffer
                buffer = session_manager.get_frame_buffer(session_id)
                keypoints_sequence = list(buffer)

                # Make prediction
                result = await predictor.predict_sequence(keypoints_sequence)

                # Update prediction count
                session = session_manager.get_session(session_id)
                if session:
                    session["prediction_count"] += 1

                # Send result
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "prediction",
                        "frame_index": data["frame_index"],
                        "result": result,
                        "session_stats": {
                            "frame_count": session["frame_count"],
                            "prediction_count": session["prediction_count"],
                        },
                    },
                )

                logger.info(f"Prediction sent to session {session_id}: {result}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
    finally:
        # Cleanup
        connection_manager.disconnect(session_id)
        session_manager.cleanup_session(session_id)
