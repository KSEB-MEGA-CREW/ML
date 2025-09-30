# Claude README - 수어 인식 F2T 서버

## 프로젝트 개요

이 프로젝트는 **수담(手談)** - 한국 수어 인식 및 번역 시스템의 **F2T (Frame-to-Text) 서버**입니다. FastAPI와 WebSocket을 사용하여 실시간 수어 인식을 제공하고, MediaPipe로 키포인트를 추출한 후 Claude API를 통해 Gloss 시퀀스를 자연어로 변환합니다.

## 프로젝트 구조

```
kseb_project_ML/
├── ml-server/                     # 메인 AI 서버
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py         # 환경 설정
│   │   │   └── dependencies.py   # 의존성 관리
│   │   ├── models/
│   │   │   ├── model_manager.py  # TensorFlow 모델 관리
│   │   │   └── predictor.py      # 예측 로직
│   │   ├── services/
│   │   │   ├── claude_service.py # Claude API 연동
│   │   │   ├── gloss_collector.py # Gloss 데이터 수집
│   │   │   ├── frame_processor.py # MediaPipe 프레임 처리 (NEW)
│   │   │   └── local_translator.py # 로컬 번역
│   │   ├── websockets/
│   │   │   ├── handlers.py       # WebSocket 핸들러
│   │   │   ├── message_types.py  # 메시지 타입 정의
│   │   │   └── session_manager.py # 세션 관리
│   │   ├── utils/
│   │   │   └── logger.py         # 로깅 유틸리티
│   │   └── main.py               # FastAPI 애플리케이션
│   ├── models/
│   │   └── label_map.json        # 라벨 매핑
│   ├── docker-compose.yml        # Docker 구성
│   ├── Dockerfile                # Docker 이미지
│   ├── requirements.txt          # Python 의존성
│   ├── .env.dev                  # 개발 환경 설정
│   └── .env.prod                 # 프로덕션 환경 설정
├── venv/                         # Python 가상환경
├── .gitignore
└── LICENSE                       # MIT 라이센스
```

## 주요 기술 스택

### F2T Server
- **FastAPI**: 비동기 웹 프레임워크
- **WebSocket**: 실시간 통신
- **MediaPipe**: 키포인트 추출 (NEW)
- **TensorFlow**: 머신러닝 모델 추론
- **Anthropic Claude**: AI 자연어 처리

### Infrastructure
- **Docker**: 컨테이너화
- **Redis**: 세션 관리 (선택사항)
- **uvicorn**: ASGI 서버

### Dependencies
```
# Core Framework
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
pydantic==2.5.0

# AI/ML
tensorflow==2.15.0
mediapipe==0.10.7      # NEW: 키포인트 추출
anthropic==0.8.1
opencv-python==4.8.1.78  # NEW: 이미지 처리
Pillow==10.1.0         # NEW: 이미지 변환

# Infrastructure
redis==5.0.1
```

## 환경 설정

### 필수 환경 변수
```bash
# Claude API
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_MODEL=claude-3-haiku-20240307

# JWT 인증
JWT_SECRET_KEY=your_jwt_secret

# 서버 설정
AI_SERVER_HOST=0.0.0.0
AI_SERVER_PORT=8000

# 백엔드 연동
BACKEND_URL=http://host.docker.internal:8080
```

## 실행 방법

### 1. Docker 실행 (권장)
```bash
cd ml-server
docker-compose up -d
```

### 2. 로컬 개발
```bash
cd ml-server
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### REST API
- `GET /` - 서버 정보
- `GET /health` - 헬스 체크

### WebSocket
- `WS /ws?token={jwt_token}` - 실시간 수어 인식

## F2T WebSocket 통신 플로우

1. **연결**: JWT 토큰으로 인증
2. **프레임 배치 전송**: 10개 Canvas 프레임을 Base64로 배치 전송 (NEW)
3. **키포인트 추출**: MediaPipe로 프레임에서 194차원 키포인트 추출 (NEW)
4. **Gloss 예측**: TensorFlow 모델로 키포인트 시퀀스에서 Gloss 예측
5. **Gloss 수집**: 고신뢰도 Gloss 수집
6. **Claude 번역**: Gloss 시퀀스를 자연어로 변환
7. **결과 반환**: 번역된 문장 반환

## 메시지 타입

### 클라이언트 → 서버 (NEW)
```json
{
  "type": "frame_batch",
  "frame_batch": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "... 총 10개 프레임"
  ],
  "batch_index": 1,
  "user_id": 123,
  "session_id": "session_abc123"
}
```

### 서버 → 클라이언트 (NEW)
```json
{
  "type": "batch_prediction_result",
  "predictions": [
    {
      "frame_index": 0,
      "gloss": "안녕",
      "confidence": 0.95,
      "success": true
    },
    {
      "frame_index": 1,
      "gloss": "하세요",
      "confidence": 0.87,
      "success": true
    }
  ],
  "batch_index": 1,
  "frames_processed": 10,
  "session_id": "session_abc123"
}
```

## 개발 가이드

### 코드 스타일
- Python: PEP 8 준수
- 비동기 프로그래밍 패턴 사용
- 타입 힌트 필수

### 테스트
```bash
pytest
```

### 로깅
- 구조화된 로깅 사용
- 로그 레벨: DEBUG, INFO, WARNING, ERROR

## 모니터링

### 헬스 체크
```bash
curl http://localhost:8000/health
```

### 로그 확인
```bash
docker-compose logs -f ai-server
```

## 트러블슈팅

### 일반적인 문제들

1. **모델 로딩 실패**
   - `models/gesture_model.h5` 파일 존재 확인
   - 파일 권한 확인

2. **Claude API 오류**
   - API 키 확인
   - 요청 제한 확인

3. **WebSocket 연결 실패**
   - JWT 토큰 유효성 확인
   - CORS 설정 확인

## 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여하기

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📊 성능 개선 (v2.1 - F2T)

### Frame-to-Text 최적화
- **프론트엔드**: MediaPipe 키포인트 → Canvas 프레임 배치 시스템
- **F2T 서버**: 프레임 배치 수신 → MediaPipe 키포인트 추출 → 모델 추론
- **성능 향상**: 전송 빈도 90% 감소, 추론 효율성 300% 향상

### 새로운 아키텍처
```
Frontend (Canvas) → F2T Server (MediaPipe) → ML Model → Claude API
     10 frames          194D keypoints       Gloss      Natural Language
```

### 마이그레이션 가이드
1. 프론트엔드에서 `sendFrameBatch()` 사용
2. F2T 서버에서 `FRAME_BATCH` 메시지 처리
3. MediaPipe 기반 키포인트 추출
4. 배치 단위 모델 추론으로 변경

---

**개발팀**: 수담(手談) 프로젝트  
**F2T 서버 버전**: v2.1  
**마지막 업데이트**: 2024년 9월