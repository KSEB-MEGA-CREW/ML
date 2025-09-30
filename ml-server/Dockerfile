# Dockerfile for F2T (Frame-to-Text) Server
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치 (MediaPipe/OpenCV 지원)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p /app/models

# 포트 노출
EXPOSE 8000

# 헬스체크 추가 (F2T 서버)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# F2T 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]