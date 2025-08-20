FROM python:3.11-alpine

WORKDIR /app

# Alpine 패키지 설치 (AI/ML 라이브러리 지원)
RUN apk update && apk upgrade && \
    apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    && rm -rf /var/cache/apk/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# 모델 디렉토리 생성 (권한 포함)
RUN mkdir -p /app/models && \
    addgroup -S appgroup && \
    adduser -S appuser -G appgroup && \
    chown -R appuser:appgroup /app

USER appuser

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]