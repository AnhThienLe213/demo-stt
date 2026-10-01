FROM python:3.12-slim

WORKDIR /app

# wget và bzip2 dùng để tải, giải nén Zipformer lúc build.
RUN apt-get update && apt-get install -y --no-install-recommends wget bzip2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py vad.py vietnamese.py index.html ./

# Zipformer-30M-RNNT-6000h — dùng đúng bản int8 do k2-fsa đóng gói sẵn cho sherpa-onnx
# (khác HF repo gốc — xem lịch sử trao đổi: bản HF repo trực tiếp là export offline
# không tương thích OnlineRecognizer; bản này là bản offline chính thức, đã kiểm chứng
# tên file khớp với pick() trong TransducerBackend).
RUN mkdir -p /app/models/zipformer && \
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2 \
    -O /tmp/zipformer.tar.bz2 && \
    tar xjf /tmp/zipformer.tar.bz2 -C /app/models/zipformer --strip-components=1 && \
    rm /tmp/zipformer.tar.bz2
ENV ZIPFORMER_MODEL_DIR=/app/models/zipformer

# Hugging Face Docker Spaces routes traffic to the configured app_port.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
