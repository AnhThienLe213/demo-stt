"""VAD Silero chạy theo dòng, giữ trạng thái LSTM giữa các khung 32ms.

`faster_whisper.vad.SileroVADModel.__call__` khởi tạo lại h/c ở mỗi lần gọi nên chỉ đúng khi
đã có trọn vẹn đoạn audio. Ở đây phải quyết định "đang nói / đã im" ngay lúc từng khung chạy
tới, nên tự lái session ONNX và mang h/c + context sang lần sau. Dùng lại đúng file model
faster-whisper đã đóng gói sẵn — không thêm phụ thuộc nào (không cần torch).

Chi phí: 1 bước LSTM cho mỗi 32ms audio, đo được ~0.3ms/khung (RTF ~0.01) nên chạy thẳng
trong vòng lặp nhận WebSocket mà không cần luồng riêng.
"""
import glob
import os

import numpy as np
import onnxruntime
from faster_whisper.vad import get_assets_path

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32ms — kích thước khung Silero v5/v6 bắt buộc ở 16kHz
CONTEXT_SAMPLES = 64
FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE


def _model_paths() -> tuple[str, ...]:
    assets = get_assets_path()
    v6_path = os.path.join(assets, "silero_vad_v6.onnx")
    if os.path.exists(v6_path):
        return ("v6", v6_path)

    encoder_path = os.path.join(assets, "silero_encoder_v5.onnx")
    decoder_path = os.path.join(assets, "silero_decoder_v5.onnx")
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        return ("v5", encoder_path, decoder_path)

    hits = sorted(glob.glob(os.path.join(assets, "*vad*.onnx")))
    if not hits:
        raise RuntimeError(f"không tìm thấy model Silero VAD trong {assets}: {os.listdir(assets)}")
    return ("legacy", hits[-1])


class StreamingVad:
    """Trả về xác suất có tiếng nói cho từng khung 512 mẫu, có nhớ ngữ cảnh."""

    def __init__(self) -> None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        paths = _model_paths()
        self.version = paths[0]
        if self.version == "v5":
            self.encoder_session = onnxruntime.InferenceSession(
                paths[1], providers=["CPUExecutionProvider"], sess_options=opts
            )
            self.decoder_session = onnxruntime.InferenceSession(
                paths[2], providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(
                paths[1], providers=["CPUExecutionProvider"], sess_options=opts
            )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"khung phải đúng {FRAME_SAMPLES} mẫu, nhận {frame.shape[0]}")
        inp = np.concatenate([self._context, frame])[None, :].astype(np.float32)
        if self.version == "v5":
            encoded = self.encoder_session.run(None, {"input": inp})[0]
            encoded = np.asarray(encoded).reshape(encoded.shape[0], -1, 128)[:, 0, :]
            probs, self._state = self.decoder_session.run(
                None, {"input": encoded, "state": self._state}
            )
        else:
            probs, self._h, self._c = self.session.run(
                None, {"input": inp, "h": self._h, "c": self._c}
            )
        self._context = frame[-CONTEXT_SAMPLES:].astype(np.float32).copy()
        return float(np.asarray(probs).reshape(-1)[-1])


class SpeechSegmenter:
    """Biến dòng xác suất VAD thành ranh giới câu, tính theo chỉ số khung.

    Hai ngưỡng (trigger / release) tạo trễ đóng mở: đã vào trạng thái nói thì chỉ thoát khi
    xác suất tụt hẳn dưới `release`, tránh nhấp nháy ở các âm nhỏ giữa câu.

    Sự kiện trả về:
      ("start", start_frame)
      ("end",   start_frame, end_frame_exclusive, reason)   reason: "silence" | "maxlen"
    """

    def __init__(
        self,
        trigger: float = 0.5,
        release: float = 0.35,
        end_silence_sec: float = 0.6,
        pad_sec: float = 0.25,
        min_speech_sec: float = 0.20,
        max_utterance_sec: float = 20.0,
    ) -> None:
        self.trigger = trigger
        self.release = release
        self.end_silence_frames = max(1, round(end_silence_sec / FRAME_SEC))
        self.pad_frames = max(0, round(pad_sec / FRAME_SEC))
        self.min_speech_frames = max(1, round(min_speech_sec / FRAME_SEC))
        self.max_utterance_frames = max(1, round(max_utterance_sec / FRAME_SEC))

        self.speaking = False
        self._n = 0  # tổng số khung đã nạp
        self._start = 0  # khung mở đầu câu hiện tại (đã trừ pad)
        self._last_voiced = 0  # khung cuối cùng còn tiếng
        self._silence = 0  # số khung im liên tiếp
        self._voiced = 0  # số khung có tiếng trong câu hiện tại

    @property
    def frames_seen(self) -> int:
        return self._n

    @property
    def utterance_start(self) -> int:
        return self._start

    def feed(self, prob: float):
        idx = self._n
        self._n += 1
        events = []

        if not self.speaking:
            if prob >= self.trigger:
                self.speaking = True
                # lùi lại `pad` khung để không cắt mất phụ âm đầu
                self._start = max(0, idx - self.pad_frames)
                self._last_voiced = idx
                self._silence = 0
                self._voiced = 1
                events.append(("start", self._start))
            return events

        # đang trong câu
        if prob >= self.release:
            self._last_voiced = idx
            self._silence = 0
            self._voiced += 1
        else:
            self._silence += 1

        if self._silence >= self.end_silence_frames:
            end = min(self._n, self._last_voiced + 1 + self.pad_frames)
            events.append(self._close(end, "silence"))
        elif idx - self._start + 1 >= self.max_utterance_frames:
            # câu quá dài: chốt tại đây và mở câu mới ngay, người dùng vẫn đang nói
            end = self._n
            events.append(self._close(end, "maxlen"))
            self.speaking = True
            self._start = end
            self._last_voiced = idx
            self._silence = 0
            self._voiced = 1
            events.append(("start", self._start))

        return events

    def flush(self):
        """Chốt câu đang dở khi hết stream."""
        if not self.speaking:
            return []
        return [self._close(self._n, "eof")]

    def _close(self, end: int, reason: str):
        start, voiced = self._start, self._voiced
        self.speaking = False
        self._silence = 0
        self._voiced = 0
        if voiced < self.min_speech_frames:
            return ("drop", start, end, reason)
        return ("end", start, end, reason)
