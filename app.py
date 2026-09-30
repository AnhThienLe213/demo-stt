"""STT tiếng Việt hai tầng: đoán nhanh theo chunk + chốt chắc theo câu.

Vì sao không dùng LocalAgreement (bản phowhisper-stream ở cổng 8891):
    LocalAgreement decode lại TOÀN BỘ buffer đang phình sau mỗi chunk rồi chốt phần prefix
    trùng nhau giữa hai lần. Chi phí vì thế tăng theo bình phương độ dài câu, và trên CPU nó
    tự dồn việc lên chính mình: đo thực tế 9.36s audio mất 29.9s mới xong (chậm hơn thời gian
    thực 3.2 lần) kể cả sau khi đã vá beam_size, condition_on_previous_text và vad_filter.
    Tệ hơn: chữ Whisper bịa ra ở các lần decode dở dang được CHỐT thẳng vào transcript.

Kiến trúc ở đây tách hẳn hai việc đó ra:

    tầng ĐOÁN (preview)   — trong lúc còn đang nói, decode lại đoạn đang nói để hiện chữ chạy.
                            Kết quả chỉ để nhìn, KHÔNG BAO GIỜ được ghép vào transcript.
                            Sai thì thôi, câu chốt đè lên.
    tầng CHỐT (confirm)   — Silero VAD phát hiện hết câu, decode trọn câu đúng một lần.
                            Đây là thứ duy nhất được tính là kết quả.

Hai luật giữ cho nó không sập như LocalAgreement:
  1. MỘT luồng decode duy nhất cho cả tiến trình. Preview không bao giờ chạy song song với
     confirm — nếu không chúng giành CPU của nhau và cả hai cùng trễ.
  2. Preview KHÔNG xếp hàng. Chỉ có đúng một ô chờ; preview mới đè lên preview cũ chưa chạy.
     Đây chính là chỗ LocalAgreement chết: nó xếp việc vào hàng đợi nên càng chạy càng tụt.
  Hệ quả: tần suất chữ chạy tự co giãn theo tải máy, còn độ trễ chốt câu thì không đổi.

Độ trễ chốt câu ≈ END_SILENCE_SEC + thời gian decode trọn câu (đo trên M4, PhoWhisper-small
int8, greedy: câu 2s ≈ 1.1s, câu 5s ≈ 1.2s, câu 9s ≈ 1.7s).

Giao thức WebSocket /ws:
    client -> nhị phân: PCM16 little-endian, mono, 16000Hz (gửi bao nhiêu byte cũng được)
    client -> text {"type":"eof"}: hết stream, chốt nốt câu đang dở
    server -> {"type":"partial","text":...,"utt":n}     chữ đoán, luôn thay thế dòng cũ
    server -> {"type":"final","text":...,"utt":n,...}   chữ đã chốt, nối vào transcript
    server -> {"type":"speech","on":true|false}         đèn báo đang nói
    server -> {"type":"dropped","utt":n}                câu quá ngắn, coi là nhiễu
"""
import asyncio
import collections
import json
import os
import re
import threading
import time
import tarfile
import tempfile
import urllib.request

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from vietnamese import looks_vietnamese
from vad import FRAME_SAMPLES, FRAME_SEC, SAMPLE_RATE, SpeechSegmenter, StreamingVad

try:
    import sherpa_onnx
except ImportError:  # chỉ bắt buộc nếu thật sự bật model transducer
    sherpa_onnx = None

load_dotenv()


def _env_f(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_i(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MODEL_SIZE = os.environ.get("PHOWHISPER_SIZE", "small")
REPO_ID = "quocphu/PhoWhisper-ct2-FasterWhisper"
SUBFOLDER = f"PhoWhisper-{MODEL_SIZE}-ct2-fasterWhisper"
LANGUAGE = os.environ.get("LANGUAGE", "vi")

# --- cắt câu ---
VAD_TRIGGER = _env_f("VAD_TRIGGER", 0.5)
VAD_RELEASE = _env_f("VAD_RELEASE", 0.35)
END_SILENCE_SEC = _env_f("END_SILENCE_SEC", 0.6)
SPEECH_PAD_SEC = _env_f("SPEECH_PAD_SEC", 0.40)
MIN_SPEECH_SEC = _env_f("MIN_SPEECH_SEC", 0.35)
MAX_UTTERANCE_SEC = _env_f("MAX_UTTERANCE_SEC", 20.0)

# --- tầng đoán ---
PREVIEW_ENABLED = os.environ.get("PREVIEW_ENABLED", "1") not in ("0", "false", "False", "")
PREVIEW_INTERVAL_SEC = _env_f("PREVIEW_INTERVAL_SEC", 0.3)
PREVIEW_MIN_SEC = _env_f("PREVIEW_MIN_SEC", 0.2)
# Chặn độ dài đoạn đem đi đoán: DecodeWorker chạy transcribe() ĐỒNG BỘ, một khi đã lấy
# job preview ra khỏi hàng đợi thì nó chạy tới hết dù ngay sau đó confirm mới đến cũng
# phải chờ (không có cơ chế huỷ giữa chừng). Preview càng dài thì worst-case confirm bị
# trễ thêm càng lâu — giá trị này chính là trần trên cho độ trễ tăng thêm đó, nên giữ nó
# ngắn hơn nhiều so với END_SILENCE_SEC để phần lớn thời gian confirm không phải chờ.
PREVIEW_MAX_SEC = _env_f("PREVIEW_MAX_SEC", 3.0)
PREVIEW_BEAM = _env_i("PREVIEW_BEAM", 1)

# --- tầng chốt ---
CONFIRM_BEAM = _env_i("CONFIRM_BEAM", 1)
# Cho phép hạ nhiệt độ dần khi bản decode đầu trông như rác (tỉ lệ nén cao = lặp câu,
# logprob thấp = model không chắc). Chỉ bật ở tầng chốt vì nó tốn thêm lượt decode.
CONFIRM_TEMPERATURES = tuple(
    float(x) for x in os.environ.get("CONFIRM_TEMPERATURES", "0.0,0.2,0.4").split(",") if x.strip()
)

# Tỉ lệ âm tiết tiếng Việt hợp lệ tối thiểu để giữ lại một đoạn chữ. PhoWhisper vẫn là Whisper
# đa ngữ fine-tune lại nên khi gặp tiếng động ngắn hay giọng không rõ, nó rơi về những chuỗi
# quen thuộc của model gốc — "of.", "you", "hello". Đo được: câu tiếng Việt thật cho tỉ lệ
# 1.00, mấy chuỗi đó cho 0.00, nên ngưỡng nằm ở đâu trong khoảng giữa cũng tách sạch.
# Đặt 0 để tắt hẳn (cần khi muốn nhận cả tên riêng nước ngoài).
VI_MIN_RATIO = _env_f("VI_MIN_RATIO", 0.6)
# Số âm tiết lạ liền nhau tối đa. Chỉ dùng tỉ lệ thì không tách được "bật wifi phòng ngủ"
# (0.75, câu Việt có từ mượn) khỏi chuỗi Whisper nghe tiếng Anh (0.67) — hai vùng chạm nhau.
VI_MAX_FOREIGN_RUN = _env_i("VI_MAX_FOREIGN_RUN", 2)
NO_SPEECH_THRESHOLD = _env_f("NO_SPEECH_THRESHOLD", 0.6)
LOG_PROB_THRESHOLD = _env_f("LOG_PROB_THRESHOLD", -1.0)
COMPRESSION_RATIO_THRESHOLD = _env_f("COMPRESSION_RATIO_THRESHOLD", 2.4)
# Trần số token sinh ra, tính theo độ dài đoạn. Đây là dây an toàn chống vòng lặp lặp chữ
# của Whisper: một câu 2.6s từng sinh ra hàng trăm dấu chấm và ngốn 11.3s decode (rtf 4.3),
# kéo cả hàng đợi phía sau. Tiếng Việt nói nhanh ~5 từ/s ≈ 12 token/s nên 16 vẫn dư biên.
MAX_TOKENS_PER_SEC = _env_f("MAX_TOKENS_PER_SEC", 16.0)
MIN_NEW_TOKENS = _env_i("MIN_NEW_TOKENS", 32)
# Đoạn do VAD cắt ra bắt đầu ngay tại chỗ có tiếng, và Whisper hay lặp âm đầu khi gặp kiểu
# vào đột ngột đó ("ch chỉnh", "m mở" — thấy đúng ở cùng một câu qua nhiều lần đo). Chèn một
# quãng im trước mặt rồi vuốt biên độ lên, đúng như khi nó nghe một file thu sẵn.
LEAD_SILENCE_SEC = _env_f("LEAD_SILENCE_SEC", 0.20)
FADE_IN_SEC = _env_f("FADE_IN_SEC", 0.01)

# Mặc định để dành 1 core cho event loop asyncio + VAD, phần còn lại (tối đa 4) cho
# decode — hardcode 4 trên máy ít core hơn sẽ oversubscribe và làm chậm mọi thứ.
_DEFAULT_CT2_THREADS = max(1, min(4, (os.cpu_count() or 4) - 1))
CT2_CPU_THREADS = _env_i("CT2_CPU_THREADS", _DEFAULT_CT2_THREADS)

app = FastAPI()
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_MODEL_ROOT = ROOT_DIR / "model"
LAB_URL = os.environ.get("LAB_URL", "http://localhost:8890")
INDEX_HTML = (
    open(os.path.join(_HERE, "index.html"), encoding="utf-8")
    .read()
    .replace("{{LAB_URL}}", LAB_URL)
    .replace("{{LAB_LABEL}}", LAB_URL.split("://", 1)[-1].rstrip("/"))
)

_local_dir = snapshot_download(repo_id=REPO_ID, allow_patterns=[f"{SUBFOLDER}/*"])
MODEL_DIR = os.environ.get("PHOWHISPER_MODEL_DIR") or os.path.join(_local_dir, SUBFOLDER)
model = WhisperModel(
    MODEL_DIR, device="cpu", compute_type="int8", cpu_threads=CT2_CPU_THREADS
)


def _warmup(m: WhisperModel) -> None:
    """Chạy một lượt decode câm lúc khởi động. CTranslate2 tốn thêm chi phí (cấp phát
    bộ nhớ, cache nội bộ) ở lần transcribe() đầu tiên trong đời tiến trình; nếu không
    làm nóng trước, client đầu tiên kết nối sẽ gánh trọn phần trễ này vào câu đầu tiên
    của họ thay vì nó được trả trước lúc server chưa nhận request nào."""
    dummy = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    try:
        list(m.transcribe(dummy, language=LANGUAGE, beam_size=1)[0])
    except Exception:
        pass


_warmup(model)


_REPEAT_PUNCT = re.compile(r"([^\w\s])\1{2,}")
_REPEAT_WORD = re.compile(r"\b(\w+)( \1\b){2,}")


def tidy(text: str) -> str:
    """Dọn dấu vết vòng lặp còn sót: dấu câu lặp và cụm từ lặp liên tiếp."""
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _REPEAT_WORD.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _lead_in(audio: np.ndarray) -> np.ndarray:
    """Thêm quãng im đầu + vuốt biên độ, chống lặp âm đầu (xem LEAD_SILENCE_SEC)."""
    n_fade = min(audio.size, round(FADE_IN_SEC * SAMPLE_RATE))
    if n_fade:
        audio = audio.copy()
        audio[:n_fade] *= np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    lead = np.zeros(round(LEAD_SILENCE_SEC * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([lead, audio])


def transcribe(audio: np.ndarray, *, beam_size: int, temperatures) -> str:
    """Decode một đoạn đã được VAD cắt sẵn.

    vad_filter tắt: đoạn vào đây đã là vùng có tiếng do Silero cắt, chạy lại VAD của
    faster-whisper chỉ tốn thêm thời gian.
    condition_on_previous_text tắt: đưa chữ cũ làm prompt khiến model bám vào lỗi trước đó và
    khuếch đại thành vòng lặp lặp câu — lỗi đã thấy rõ ở bản LocalAgreement.
    """
    dur = audio.size / SAMPLE_RATE
    segments, _info = model.transcribe(
        _lead_in(audio),
        language=LANGUAGE,
        beam_size=beam_size,
        temperature=temperatures,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        max_new_tokens=max(MIN_NEW_TOKENS, int(dur * MAX_TOKENS_PER_SEC)),
    )
    parts = []
    for seg in segments:
        # Whisper hay "chép" cả khoảng lặng thành câu chào/quảng cáo. Bỏ đoạn mà chính model
        # cũng cho là không có tiếng nói.
        if seg.no_speech_prob is not None and seg.no_speech_prob > NO_SPEECH_THRESHOLD:
            continue
        # Lọc từng đoạn trước: một câu dài có thể chỉ rác ở một khúc.
        if not looks_vietnamese(seg.text, VI_MIN_RATIO, VI_MAX_FOREIGN_RUN):
            continue
        parts.append(seg.text)
    text = tidy("".join(parts))
    # Lọc lại toàn câu: từng đoạn có thể vừa đủ qua ngưỡng mà gộp lại thì không.
    if text and not looks_vietnamese(text, VI_MIN_RATIO, VI_MAX_FOREIGN_RUN):
        return ""
    return text


class PhoWhisperBackend:
    """Bọc lại model + hàm transcribe() sẵn có, khớp interface chung `.transcribe(...)`."""

    name = "phowhisper"
    label = "PhoWhisper (Whisper fine-tune, CTranslate2)"

    def transcribe(self, audio: np.ndarray, *, beam_size: int, temperatures) -> str:
        return transcribe(audio, beam_size=beam_size, temperatures=temperatures)


class TransducerBackend:
    """gipformer-65M-rnnt / Zipformer-30M-RNNT-6000h — RNN-Transducer qua sherpa-onnx.

    Cả hai đều là bản ONNX OFFLINE (xem lịch sử trao đổi: lỗi "encoder_dims does not
    exist" khi thử OnlineRecognizer) — không streaming incremental thật, nhưng vẫn dùng
    được với đúng cơ chế "VAD cắt câu rồi decode trọn câu" hiện có của app này, kể cả cho
    preview (chỉ là decode một đoạn ngắn hơn, không có gì đặc biệt cho streaming).
    beam_size/temperatures của tham số gọi chung bị bỏ qua vì recognizer greedy cố định.
    """

    def __init__(self, name: str, label: str, model_dir: str) -> None:
        if sherpa_onnx is None:
            raise RuntimeError("chưa cài sherpa-onnx (pip install sherpa-onnx)")
        self.name = name
        self.label = label
        d = Path(model_dir)

        def pick(stem: str) -> str:
            candidates = []
            for base in [stem, f"{stem}-epoch-20-avg-10", f"{stem}-epoch-20"]:
                candidates.extend([
                    d / f"{base}.int8.onnx",
                    d / f"{base}.onnx",
                ])
            for path in candidates:
                if path.exists():
                    return str(path)
            raise FileNotFoundError(f"không tìm thấy {stem}(.int8).onnx hoặc biến thể epoch trong {d}")

        def pick_tokens() -> str:
            for candidate in ("tokens.txt", "config.json", "token.txt"):
                path = d / candidate
                if path.exists():
                    return str(path)
            raise FileNotFoundError(f"không tìm thấy tokens.txt/config.json trong {d}")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=pick("encoder"),
            decoder=pick("decoder"),
            joiner=pick("joiner"),
            tokens=pick_tokens(),
            num_threads=CT2_CPU_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        # warm-up — tránh request đầu tiên gánh chi phí khởi tạo (giống lý do đã làm cho
        # PhoWhisper ở _warmup()).
        s = self._recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, np.zeros(8000, dtype=np.float32))
        self._recognizer.decode_stream(s)

    def transcribe(self, audio: np.ndarray, *, beam_size=None, temperatures=None) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        # 2 model này không có bộ lọc "có phải tiếng Việt không" riêng như transcribe()
        # của PhoWhisper — tận dụng lại đúng bộ lọc đã có thay vì viết mới.
        if text and not looks_vietnamese(text, VI_MIN_RATIO, VI_MAX_FOREIGN_RUN):
            return ""
        return tidy(text)


def _local_model_dir(name: str) -> str:
    env_name = f"{name.upper()}_MODEL_DIR"
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    candidate = LOCAL_MODEL_ROOT / name
    if candidate.exists():
        return str(candidate)
    return str(candidate)


def _download_zipformer() -> str:
    local_dir = Path(_local_model_dir("zipformer"))
    if (local_dir / "encoder-epoch-20-avg-10.onnx").exists() or (local_dir / "encoder.onnx").exists():
        return str(local_dir)

    model_dir = Path(tempfile.gettempdir()) / "zipformer-vi-30M-int8"
    if (model_dir / "encoder.onnx").exists():
        return str(model_dir)

    archive_path = model_dir.with_suffix(".tar.bz2")
    model_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
    )
    urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            destination = (model_dir / Path(*parts)).resolve()
            if model_dir.resolve() not in destination.parents:
                raise ValueError(f"đường dẫn không an toàn trong model archive: {member.name}")
            member.name = str(Path(*parts))
            archive.extract(member, model_dir)
    archive_path.unlink(missing_ok=True)
    return str(model_dir)


def _build_backends() -> dict:
    """Nạp các model còn lại khi app khởi động, bỏ qua backend nào tải/nạp lỗi."""
    backends = {"phowhisper": PhoWhisperBackend()}
    specs = [
        (
            "gipformer",
            "Gipformer 65M RNN-T",
            lambda: _local_model_dir("gipformer"),
        ),
        (
            "zipformer",
            "Zipformer 30M RNN-T",
            lambda: _local_model_dir("zipformer"),
        ),
    ]
    for name, label, get_model_dir in specs:
        try:
            model_dir = get_model_dir()
            backends[name] = TransducerBackend(name, label, model_dir)
        except Exception as exc:  # noqa: BLE001 - thiếu 1 model không được làm chết server
            print(f"[!] Bỏ qua model '{name}': {exc}")
    return backends


class _Job:
    __slots__ = ("audio", "backend", "beam", "temps", "kind", "utt", "done", "stale")

    def __init__(self, audio, backend, beam, temps, kind, utt, done, stale):
        self.audio = audio
        self.backend = backend
        self.beam = beam
        self.temps = temps
        self.kind = kind
        self.utt = utt
        self.done = done
        self.stale = stale


class DecodeWorker:
    """Một luồng decode duy nhất cho cả tiến trình, confirm được ưu tiên tuyệt đối.

    Preview chỉ có đúng MỘT ô chờ (`_preview`): cái mới đè cái cũ chưa kịp chạy. Không có
    hàng đợi preview, nên máy chậm đi thì chữ chạy thưa ra chứ không bao giờ tụt hậu tích luỹ.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._confirms = collections.deque()
        self._preview = None
        self._thread = threading.Thread(target=self._run, name="decode", daemon=True)
        self._thread.start()

    def submit_confirm(self, job: _Job) -> None:
        with self._cv:
            self._confirms.append(job)
            self._cv.notify()

    def offer_preview(self, job: _Job) -> None:
        with self._cv:
            if self._confirms:
                return  # có câu đang chờ chốt thì đừng chen vào
            self._preview = job
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._confirms and self._preview is None:
                    self._cv.wait()
                if self._confirms:
                    job = self._confirms.popleft()
                else:
                    job, self._preview = self._preview, None
            if job.stale():
                continue
            started = time.perf_counter()
            try:
                text = job.backend.transcribe(job.audio, beam_size=job.beam, temperatures=job.temps)
                err = None
            except Exception as exc:  # decode hỏng không được giết luồng dùng chung
                text, err = "", repr(exc)
            job.done(job, text, err, time.perf_counter() - started)


worker = DecodeWorker()
BACKENDS = _build_backends()
DEFAULT_BACKEND = "phowhisper"


class Session:
    """Trạng thái của một kết nối WebSocket."""

    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop, backend_name: str) -> None:
        self.ws = ws
        self.loop = loop
        self.backend = BACKENDS[backend_name]
        self.out: asyncio.Queue = asyncio.Queue()
        self.vad = StreamingVad()
        self.seg = SpeechSegmenter(
            trigger=VAD_TRIGGER,
            release=VAD_RELEASE,
            end_silence_sec=END_SILENCE_SEC,
            pad_sec=SPEECH_PAD_SEC,
            min_speech_sec=MIN_SPEECH_SEC,
            max_utterance_sec=MAX_UTTERANCE_SEC,
        )
        self._tail = np.zeros(0, dtype=np.float32)  # mẫu lẻ chưa đủ một khung
        # Giữ lại audio ngay trước lúc VAD kích hoạt để không cắt mất phụ âm đầu.
        self._ring = collections.deque(maxlen=round((SPEECH_PAD_SEC + 1.0) / FRAME_SEC) + 2)
        self._cur = -1  # chỉ số khung vừa nạp
        self._utt_frames: list | None = None
        self._utt_start = 0
        self.utt_id = 0
        self._preview_at = 0  # chỉ số khung của lần đoán gần nhất
        self._pending = 0  # số câu đang chờ decode
        self._eof = False
        self.closed = False

    def emit(self, msg: dict) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, msg)

    # ---------- nạp audio ----------
    def feed_pcm(self, raw: bytes) -> None:
        if len(raw) % 2:
            raw = raw[:-1]
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if self._tail.size:
            pcm = np.concatenate([self._tail, pcm])
        n_full = pcm.size // FRAME_SAMPLES
        self._tail = pcm[n_full * FRAME_SAMPLES:].copy()
        for i in range(n_full):
            self._frame(pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
        if n_full:
            self._maybe_preview()

    def _frame(self, frame: np.ndarray) -> None:
        self._cur = self.seg.frames_seen
        self._ring.append(frame)
        if self._utt_frames is not None:
            self._utt_frames.append(frame)
        for ev in self.seg.feed(self.vad(frame)):
            self._event(ev)

    @property
    def _ring_start(self) -> int:
        """Chỉ số khung của phần tử đầu ring (ring luôn kết thúc ở khung self._cur)."""
        return self._cur - len(self._ring) + 1

    # ---------- ranh giới câu ----------
    def _event(self, ev) -> None:
        if ev[0] == "start":
            self._open(ev[1])
            return

        kind, start, end, reason = ev
        frames = self._utt_frames or []
        keep = max(0, end - self._utt_start)
        chunk = frames[:keep]
        utt = self.utt_id
        self._utt_frames = None
        self.emit({"type": "speech", "on": False, "utt": utt})

        if kind == "drop" or not chunk:
            self.emit({"type": "dropped", "utt": utt, "reason": reason})
        else:
            self._confirm(np.concatenate(chunk), utt, reason)

        if reason == "maxlen":
            # segmenter đã mở câu mới ngay lập tức (người dùng vẫn đang nói)
            self._open(self.seg.utterance_start)

    def _open(self, start: int) -> None:
        # kéo lại phần pad còn nằm trong ring
        offset = max(0, start - self._ring_start)
        self._utt_frames = list(self._ring)[offset:]
        self._utt_start = max(start, self._ring_start)
        self.utt_id += 1
        self._preview_at = self.seg.frames_seen
        self.emit({"type": "speech", "on": True, "utt": self.utt_id})

    # ---------- tầng chốt ----------
    def _confirm(self, audio: np.ndarray, utt: int, reason: str) -> None:
        dur = audio.size / SAMPLE_RATE
        self._pending += 1

        def done(job, text, err, secs):
            self.emit({
                "type": "final",
                "utt": job.utt,
                "text": text,
                "error": err,
                "audio_sec": round(dur, 2),
                "decode_sec": round(secs, 2),
                "rtf": round(secs / dur, 3) if dur else None,
                "reason": reason,
            })
            self.loop.call_soon_threadsafe(self._settle)

        worker.submit_confirm(
            _Job(audio, self.backend, CONFIRM_BEAM, CONFIRM_TEMPERATURES, "confirm", utt, done,
                 lambda: self.closed)
        )

    def _settle(self) -> None:
        """Chạy trên event loop: một câu vừa chốt xong."""
        self._pending -= 1
        if self._eof and self._pending <= 0:
            self.out.put_nowait({"type": "eof_ack"})

    # ---------- tầng đoán ----------
    def _maybe_preview(self) -> None:
        if not PREVIEW_ENABLED or self._utt_frames is None:
            return
        # Nhịp tính theo thời lượng audio đã nạp, không theo đồng hồ tường: như vậy hành vi
        # giống nhau dù stream chạy đúng thời gian thực hay tua nhanh khi chạy test.
        if (self.seg.frames_seen - self._preview_at) * FRAME_SEC < PREVIEW_INTERVAL_SEC:
            return
        if len(self._utt_frames) * FRAME_SEC < PREVIEW_MIN_SEC:
            return
        self._preview_at = self.seg.frames_seen

        keep = round(PREVIEW_MAX_SEC / FRAME_SEC)
        frames = self._utt_frames[-keep:]  # câu dài thì chỉ đoán phần đuôi
        tail_only = len(frames) < len(self._utt_frames)
        audio = np.concatenate(frames)
        utt = self.utt_id

        def done(job, text, err, secs):
            if err or not text:
                return
            self.emit({
                "type": "partial",
                "utt": job.utt,
                "text": text,
                "tail_only": tail_only,
                "decode_sec": round(secs, 2),
            })

        worker.offer_preview(
            _Job(audio, self.backend, PREVIEW_BEAM, (0.0,), "preview", utt, done,
                 # Câu đã chốt xong thì bản đoán vô nghĩa. Đây là chốt chặn bảo đảm chữ bịa ở
                 # tầng đoán không bao giờ chạm được vào transcript.
                 lambda u=utt: self.closed or self.utt_id != u or self._utt_frames is None)
        )

    def flush(self) -> None:
        self._eof = True
        for ev in self.seg.flush():
            self._event(ev)
        if self._pending <= 0:
            self.out.put_nowait({"type": "eof_ack"})


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/models")
def list_models():
    return {
        "default": DEFAULT_BACKEND,
        "models": [{"name": b.name, "label": b.label} for b in BACKENDS.values()],
    }


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model": SUBFOLDER,
        "model_dir": MODEL_DIR,
        "models_loaded": list(BACKENDS.keys()),
        "sample_rate": SAMPLE_RATE,
        "config": {
            "vad_trigger": VAD_TRIGGER,
            "vad_release": VAD_RELEASE,
            "end_silence_sec": END_SILENCE_SEC,
            "speech_pad_sec": SPEECH_PAD_SEC,
            "max_utterance_sec": MAX_UTTERANCE_SEC,
            "preview_enabled": PREVIEW_ENABLED,
            "preview_interval_sec": PREVIEW_INTERVAL_SEC,
            "preview_min_sec": PREVIEW_MIN_SEC,
            "preview_max_sec": PREVIEW_MAX_SEC,
            "preview_beam": PREVIEW_BEAM,
            "confirm_beam": CONFIRM_BEAM,
            "confirm_temperatures": list(CONFIRM_TEMPERATURES),
            "max_tokens_per_sec": MAX_TOKENS_PER_SEC,
            "lead_silence_sec": LEAD_SILENCE_SEC,
            "vi_min_ratio": VI_MIN_RATIO,
            "vi_max_foreign_run": VI_MAX_FOREIGN_RUN,
            "min_speech_sec": MIN_SPEECH_SEC,
            "cpu_threads": CT2_CPU_THREADS,
        },
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    requested = ws.query_params.get("model", DEFAULT_BACKEND)
    backend_name = requested if requested in BACKENDS else DEFAULT_BACKEND
    sess = Session(ws, asyncio.get_running_loop(), backend_name)

    async def pump():
        while True:
            await ws.send_text(json.dumps(await sess.out.get(), ensure_ascii=False))

    pumper = asyncio.create_task(pump())
    try:
        await ws.send_text(json.dumps({
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "model": backend_name,
            # client xin model không tồn tại/chưa nạp được thì báo lại đã fallback về gì,
            # để UI có thể tự cập nhật dropdown thay vì âm thầm dùng nhầm model.
            "requested_model": requested,
        }))
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                sess.feed_pcm(msg["bytes"])
            elif msg.get("text"):
                try:
                    cmd = json.loads(msg["text"])
                except ValueError:
                    continue
                if cmd.get("type") == "eof":
                    sess.flush()
    except WebSocketDisconnect:
        pass
    finally:
        sess.closed = True
        # nhường vài vòng cho pump đẩy nốt những gì đã xếp hàng trước khi đóng
        for _ in range(3):
            await asyncio.sleep(0)
        pumper.cancel()
