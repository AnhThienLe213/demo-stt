FROM python:3.12-slim

WORKDIR /app

# ffmpeg không bắt buộc cho pipeline PCM16 hiện tại, nhưng hay cần nếu sau này nhận
# thêm định dạng audio khác từ trình duyệt. wget để tải sẵn model Zipformer lúc build.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget bzip2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py vad.py vietnamese.py index.html ./

# Tải sẵn model vào image lúc build — tránh việc mỗi lần cold start (scale-to-zero)
# phải tải lại từ Hugging Face, vốn cộng thêm hàng chục giây vào lần request đầu tiên.
ENV HF_HOME=/app/.cache/huggingface
ENV PHOWHISPER_SIZE=small
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='quocphu/PhoWhisper-ct2-FasterWhisper', \
    allow_patterns=['PhoWhisper-small-ct2-fasterWhisper/*'])"

# gipformer-65M-rnnt — chỉ tải file .onnx + tokens.txt, bỏ qua checkpoint PyTorch nặng
# (repo có cả 2 vì hỗ trợ infer_pytorch.py cho việc fine-tune, không cần cho serving).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='g-group-ai-lab/gipformer-65M-rnnt', \
    local_dir='/app/models/gipformer', \
    allow_patterns=['*.onnx', 'tokens.txt'])"
ENV GIPFORMER_MODEL_DIR=/app/models/gipformer

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
