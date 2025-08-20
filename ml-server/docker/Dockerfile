FROM python:3.11-alpine

WORKDIR /app

# Alpine에서 ML/AI 라이브러리에 필요한 패키지들 설치
RUN apk update && apk upgrade && \
    apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    lapack-dev \
    jpeg-dev \
    zlib-dev \
    freetype-dev \
    lcms2-dev \
    openjpeg-dev \
    tiff-dev \
    tk-dev \
    tcl-dev \
    harfbuzz-dev \
    fribidi-dev \
    libc-dev \
    && rm -rf /var/cache/apk/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 나머지 설정...
RUN mkdir -p /app/models && \
    addgroup -S appgroup && \
    adduser -S appuser -G appgroup && \
    chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]