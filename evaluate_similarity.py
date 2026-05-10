import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()

# 내 목소리 기준 (프롬프트 중 하나를 레퍼런스로)
reference_path = "input/prompt_1.wav"

# 비교할 VC 결과물들
output_dir = Path("output")
output_files = sorted(output_dir.glob("scene2_*.wav"))

ref_wav = preprocess_wav(reference_path)
ref_embed = encoder.embed_utterance(ref_wav)

print(f"기준 음성: {reference_path}\n")
scores = []
for out_file in output_files:
    wav = preprocess_wav(str(out_file))
    embed = encoder.embed_utterance(wav)
    score = np.dot(ref_embed, embed)
    scores.append((score, out_file.name))
    print(f"  {out_file.name}: {score:.4f}")

best_score, best_file = max(scores)
print(f"\n가장 유사한 결과물: {best_file} (score: {best_score:.4f})")
