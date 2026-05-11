"""
원본 영상 + VC 클로닝 오디오를 합성해서 최종 영상을 만드는 스크립트.

원리:
  - 클로닝 구간: 원본 오디오 묵음 + VC 음성만 출력
  - 그 외 구간: 원본 오디오(남자 대사·BGM) 그대로 유지
"""
import subprocess
import sys
import os

# ── 설정 ──────────────────────────────────────────────
URL = "https://www.youtube.com/watch?v=n3fOof8xXR8"
START = "00:01:31"
END = "00:02:00"

# 원본 영상 타임스탬프 기준 (초 단위) — preprocess_audio.py와 동일하게 맞출 것
KEEP_SEGMENTS_ABS = [
    (91.3, 101),
    (104.3, 120),
]

VC_AUDIO  = "output/scene3_final_prompt.wav"
FINAL_OUT = "output_video/scene3_final_prompt.mp4"

# 클로닝 구간 경계의 페이드 길이 (초) — 높이면 전환이 더 부드럽고 자연스러움
FADE_SEC = 0.15
# ─────────────────────────────────────────────────────

VIDEO_FULL = "input/video_full.mp4"
VIDEO_CUT  = "input/video_cut.mp4"


def time_to_sec(t):
    parts = t.split(":")
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(parts)))


def make_vol_expr(segments, fade=FADE_SEC):
    """경계마다 페이드를 적용한 ffmpeg volume 표현식 생성.
    - [s-fade, s]: 1→0 페이드아웃 (원본 서서히 묵음)
    - [s,     e]: 0  (묵음, VC 음성 재생)
    - [e, e+fade]: 0→1 페이드인  (원본 서서히 복귀)
    """
    def r(v):
        return round(v, 3)

    exprs = []
    for s, e in segments:
        s0 = r(max(0.0, s - fade))
        s, e, f = r(s), r(e), r(fade)
        exprs.append(
            f"if(between(t,{s0},{s}),({s}-t)/{f},"
            f"if(between(t,{s},{e}),0,"
            f"if(between(t,{e},{r(e+fade)}),(t-{e})/{f},1)))"
        )
    # ffmpeg min()은 인수 2개만 지원 → 중첩으로 처리
    result = exprs[0]
    for expr in exprs[1:]:
        result = f"min({result},{expr})"
    return result


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)


os.makedirs("output", exist_ok=True)
os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)

# 영상 다운로드 (화질 최고 mp4)
run([
    "yt-dlp",
    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
    "-o", VIDEO_FULL,
    "--force-overwrites",
    URL,
])

# 정밀 컷
run([
    "ffmpeg", "-y",
    "-i", VIDEO_FULL,
    "-ss", START, "-to", END,
    "-c:v", "libx264", "-c:a", "aac",
    VIDEO_CUT,
])

# 클로닝 구간에서만 원본 오디오 묵음 처리하는 ffmpeg 볼륨 표현식 생성
start_sec = time_to_sec(START)
keep_rel = [(abs_s - start_sec, abs_e - start_sec) for abs_s, abs_e in KEEP_SEGMENTS_ABS]
vol_filter = f"volume='{make_vol_expr(keep_rel)}':eval=frame"

run([
    "ffmpeg", "-y",
    "-i", VIDEO_CUT,
    "-i", VC_AUDIO,
    "-filter_complex",
    f"[0:a]{vol_filter}[muted];[muted][1:a]amix=inputs=2:duration=first:normalize=0",
    "-map", "0:v",
    "-c:v", "copy",
    "-c:a", "aac",
    FINAL_OUT,
])

os.remove(VIDEO_FULL)
os.remove(VIDEO_CUT)

print(f"\n완료: {FINAL_OUT}")
