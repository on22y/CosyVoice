import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import torch
import soundfile as sf
import os

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

source_audio = 'input/source_audio_scene2.wav'

target_voices = [f'input/prompt_{i}.wav' for i in range(1, 6)]

CHUNK_SEC = 28  # 30초 제한보다 여유 있게 설정
os.makedirs('output', exist_ok=True)


def load_wav(path):
    audio, sr = sf.read(path, always_2d=True)  # (time, channels)
    wav = torch.from_numpy(audio.T).float()    # (channels, time)
    if sr != 16000:
        import julius
        wav = julius.resample_frac(wav, sr, 16000)
    return wav  # (channels, time)


def split_chunks(wav, chunk_samples):
    total = wav.shape[1]
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunks.append(wav[:, start:end])
        start = end
    return chunks


for prompt_num, target_voice in enumerate(target_voices, start=1):
    source_wav = load_wav(source_audio)
    chunk_samples = CHUNK_SEC * 16000
    chunks = split_chunks(source_wav, chunk_samples)

    all_output = []
    for chunk_idx, chunk in enumerate(chunks):
        tmp_chunk = f'input/_chunk_{chunk_idx}.wav'
        sf.write(tmp_chunk, chunk.squeeze(0).numpy(), 16000)

        for output in cosyvoice.inference_vc(tmp_chunk, target_voice):
            all_output.append(output['tts_speech'])

        os.remove(tmp_chunk)

    merged = torch.cat(all_output, dim=-1)
    out_path = f'output/scene2_{prompt_num}.wav'
    torchaudio.save(out_path, merged, cosyvoice.sample_rate)
    print(f'저장 완료: {out_path}')

print('done')
