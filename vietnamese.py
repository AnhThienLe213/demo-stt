"""Nhận ra một chuỗi có phải tiếng Việt không, dựa vào cấu trúc âm tiết.

Vì sao cần: PhoWhisper vẫn là Whisper đa ngữ fine-tune lại, nên khi gặp tiếng động ngắn, hơi
thở hay giọng không rõ, nó rơi về những chuỗi hay gặp nhất của model gốc — "of.", "you",
"hello", "Thank you." Đây là chỗ chữ rác lọt vào transcript dù VAD đã cắt đúng.

Vì sao không dùng từ điển: âm tiết tiếng Việt sinh ra theo luật, không phải danh sách đóng. Một
âm tiết hợp lệ = (âm đầu)(âm chính)(âm cuối), mỗi phần lấy từ tập hữu hạn. Kiểm theo luật thì
bắt được cả từ hiếm lẫn tên riêng thuần Việt, mà không cần mang theo file từ điển nào.

Ví dụ: "người" = ng + ươi + ∅ hợp lệ; "hello" có 'll' nằm giữa hai nguyên âm — tiếng Việt không
có cấu trúc đó; "of" có 'f' — không thuộc bảng chữ cái tiếng Việt.
"""
import re
import unicodedata

# Dấu thanh — bỏ đi trước khi đối chiếu. Giữ lại dấu tạo CHỮ (â ă ê ô ơ ư), vì chúng là chữ cái
# riêng chứ không phải thanh điệu.
_TONES = {"̀", "́", "̃", "̉", "̣"}

_ONSETS = (
    "ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x", "",
)
_CODAS = ("ngh", "ng", "nh", "ch", "c", "m", "n", "p", "t", "")
_NUCLEI = {
    # nguyên âm đơn
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    # nguyên âm đôi
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu", "oa", "oă", "oe",
    "oi", "oo", "ôi", "ơi", "ua", "uâ", "uă", "ue", "uê", "ui", "uô", "uơ", "uy", "ưa",
    "ưi", "ươ", "ưu", "ya", "yê", "yu",
    # nguyên âm ba
    "iêu", "oai", "oao", "oay", "oeo", "uao", "uay", "uây", "uôi", "uya", "uyê", "uyu",
    "ươi", "ươu", "yêu",
}
_VI_LETTERS = set("aăâbcdđeêghiklmnoôơpqrstuưvxy")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def strip_tone(s: str) -> str:
    """Bỏ dấu thanh, giữ nguyên chữ cái (â ă ê ô ơ ư đ)."""
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", s) if c not in _TONES)
    )


def is_syllable(word: str) -> bool:
    """Một âm tiết tiếng Việt viết đúng chính tả?"""
    w = strip_tone(word.lower())
    if not w or not set(w) <= _VI_LETTERS:
        return False  # chứa f, j, w, z hoặc ký tự lạ
    for onset in _ONSETS:  # xếp dài trước ngắn, nhưng vẫn thử hết vì "gì" = g + i
        if not w.startswith(onset):
            continue
        rest = w[len(onset):]
        for coda in _CODAS:
            if coda and not rest.endswith(coda):
                continue
            nucleus = rest[: len(rest) - len(coda)] if coda else rest
            if nucleus in _NUCLEI:
                return True
    return False


def vietnamese_ratio(text: str) -> float:
    """Tỉ lệ âm tiết hợp lệ trong chuỗi. Chuỗi không có chữ nào (chỉ số, dấu câu) tính là 1.0."""
    words = _WORD.findall(text)
    if not words:
        return 1.0
    return sum(is_syllable(w) for w in words) / len(words)


def longest_foreign_run(text: str) -> int:
    """Số âm tiết lạ NẰM LIỀN NHAU dài nhất.

    Chỉ nhìn tỉ lệ thì không đủ: "bật wifi phòng ngủ" (câu Việt có từ mượn) cho 0.75, còn
    "hello hello their how ui i am tét từ thế mic phong" (Whisper nghe tiếng Anh) cho 0.67 —
    hai vùng gần như chạm nhau. Nhưng chúng khác hẳn nhau ở hình dạng: từ mượn đứng lẻ giữa câu
    Việt, còn tiếng nước ngoài thật thì rác đi thành cụm. Đếm cụm dài nhất tách được sạch.
    """
    run = best = 0
    for w in _WORD.findall(text):
        run = 0 if is_syllable(w) else run + 1
        best = max(best, run)
    return best


def looks_vietnamese(text: str, min_ratio: float = 0.6, max_run: int = 2) -> bool:
    """Chuỗi này có đáng coi là tiếng Việt không (đủ tỉ lệ VÀ không có cụm lạ dài)."""
    if min_ratio <= 0:
        return True
    return vietnamese_ratio(text) >= min_ratio and longest_foreign_run(text) <= max_run
