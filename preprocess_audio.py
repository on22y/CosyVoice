import subprocess
import sys
import os
import soundfile as sf
import noisereduce as nr
import numpy as np
import torch

URL = "https://www.youtube.com/watch?v=JjlmxYUgwo8"
START = "00:00:56"
END = "00:01:43"
RAW_FULL = "input/source_full.wav"
RAW_OUT = "input/source_raw.wav"
FINAL_OUT = "input/source_audio.wav"
SAMPLE_RATE = 16000

# 원본 영상 타임스탬프 기준 (초 단위)
KEEP_SEGMENTS_ABS = [
    (56.5, 59.2),
    (60.7, 64),
    (66.2, 95),
    (100, 103),
]


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)


os.makedirs("input", exist_ok=True)

# 전체 오디오 다운로드 후 ffmpeg으로 정밀 컷 (키프레임 오차 방지)
run([
    "yt-dlp",
    "-x", "--audio-format", "wav",
    "--audio-quality", "0",
    "-o", RAW_FULL,
    "--force-overwrites",
    URL,
])
run([
    "ffmpeg", "-y",
    "-i", RAW_FULL,
    "-ss", START, "-to", END,
    "-c", "copy",
    RAW_OUT,
])

# demucs 라이브러리로 직접 보컬 분리
print("[RUN] demucs 보컬 분리 중...")
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model("htdemucs")
model.eval()

audio_np, sr = sf.read(RAW_OUT, always_2d=True)  # (time, channels)
wav = torch.from_numpy(audio_np.T).float()        # (channels, time)

# demucs 모델 입력 샘플레이트로 리샘플
if sr != model.samplerate:
    import julius
    wav = julius.resample_frac(wav, sr, model.samplerate)
    sr = model.samplerate

# 정규화
ref = wav.mean(0)
wav_norm = (wav - ref.mean()) / ref.std()

with torch.no_grad():
    sources = apply_model(model, wav_norm[None], device="cpu", progress=True)[0]

sources = sources * ref.std() + ref.mean()

vocals_idx = model.sources.index("vocals")
vocals = sources[vocals_idx]  # shape: (channels, time)

# soundfile로 저장
vocals_np = vocals.T.numpy()  # (time, channels)
tmp_vocals = "input/source_vocals_tmp.wav"
sf.write(tmp_vocals, vocals_np, sr)
print(f"[OK] 보컬 분리 완료: {tmp_vocals}")

# noisereduce로 잔여 잡음 제거
audio, _ = sf.read(tmp_vocals)
if audio.ndim == 2:
    audio = audio.mean(axis=1)

reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False)

# 추출할 대사 외 구간 묵음 처리
start_sec = sum(int(x) * 60 ** i for i, x in enumerate(reversed(START.split(":"))))
muted = np.zeros_like(reduced)
for abs_s, abs_e in KEEP_SEGMENTS_ABS:
    s = max(0, int((abs_s - start_sec) * sr))
    e = min(len(reduced), int((abs_e - start_sec) * sr))
    muted[s:e] = reduced[s:e]
reduced = muted

# ffmpeg으로 16kHz mono WAV 변환
tmp_nr = "input/source_nr_tmp.wav"
sf.write(tmp_nr, reduced, sr)

run([
    "ffmpeg", "-y",
    "-i", tmp_nr,
    "-ar", str(SAMPLE_RATE),
    "-ac", "1",
    FINAL_OUT,
])

os.remove(tmp_vocals)
os.remove(tmp_nr)
os.remove(RAW_FULL)
os.remove(RAW_OUT)

print(f"\n완료: {FINAL_OUT}")
print(f"voicecloning.py의 source_audio = '{FINAL_OUT}' 으로 설정")