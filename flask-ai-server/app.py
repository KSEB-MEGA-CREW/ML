# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import json
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # CORS 허용


@app.route("/analyze-frame", methods=["POST"])
def analyze_frame():
    try:
        # 요청 데이터 파싱
        data = request.get_json()

        # 백엔드에서 전송하는 데이터 구조 확인
        request_id = data.get("request_id")
        frame_data = data.get("frame_data")
        session_id = data.get("session_id")
        frame_index = data.get("frame_index")
        timestamp = data.get("timestamp")
        user_id = data.get("user_id")

        print(f"[{datetime.now()}] 프레임 수신:")
        print(f"  - Request ID: {request_id}")
        print(f"  - Session ID: {session_id}")
        print(f"  - Frame Index: {frame_index}")
        print(f"  - User ID: {user_id}")
        print(f"  - Frame Size: {len(frame_data)} bytes")

        # Base64 이미지 검증
        if not frame_data:
            return jsonify({"success": False, "error": "Frame data is empty"}), 400

        # 간단한 이미지 디코딩 테스트
        try:
            decoded_data = base64.b64decode(frame_data)
            print(f"  - Decoded Size: {len(decoded_data)} bytes")
        except Exception as e:
            return (
                jsonify({"success": False, "error": f"Invalid base64 data: {str(e)}"}),
                400,
            )

        # 시뮬레이션된 AI 처리 시간
        # time.sleep(0.1)  # 100ms 처리 시간

        # 성공 응답 (실제 AI 결과 대신 테스트 데이터)
        response = {
            "success": True,
            "request_id": request_id,
            "processing_time": 100,
            "analysis_result": {
                "detected_signs": ["안녕하세요", "감사합니다"],
                "confidence": 0.85,
                "frame_index": frame_index,
            },
        }

        print(f"[{datetime.now()}] 처리 완료.")

        return jsonify(response), 200

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "AI Frame Analysis Server",
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
