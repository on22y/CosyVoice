import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import os

def run_experiments():
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    # 프롬프트 음성
    wav_path = './asset/myvoice_2.wav'
    spk_id = 'my_korean_spk'

    # 텍스트 실험
    texts = {
        "short": "안녕하세요. 오늘 날씨가 좋네요.<|endofprompt|>",
        "medium": "안녕하세요. 오늘 날씨가 참 좋네요. 산책하기 좋은 날이에요.<|endofprompt|>",
        "long": (
            "안녕하세요. 오늘 날씨가 참 좋네요. 이렇게 좋은 날에는 산책하기 딱 좋은 것 같아요. "
            "저는 요즘 독서를 많이 하고 있는데, 여러분은 어떤 취미를 즐기시나요?<|endofprompt|>"
        ),
    }

    # Instruct 실험
    instructs = {
        "neutral": "You are a helpful assistant.<|endofprompt|>",
        "friendly": "You are a helpful assistant. Speak in a friendly tone.<|endofprompt|>",
        "slow": "You are a helpful assistant. Speak slowly and clearly.<|endofprompt|>",
        "fast": "You are a helpful assistant. Speak quickly.<|endofprompt|>",
        "cute": "You are a helpful assistant. Speak like a cute child.<|endofprompt|>",
    }

    # Speed 실험
    speeds = [0.8, 1.0, 1.2]

    # Zero-shot speaker 등록 (1회만)
    cosyvoice.add_zero_shot_spk(texts["long"], wav_path, spk_id)
    cosyvoice.save_spkinfo()

    os.makedirs("outputs", exist_ok=True)

    # 실험 루프
    for text_name, text in texts.items():
        for instruct_name, instruct in instructs.items():
            for speed in speeds:

                filename = f"outputs/{text_name}_{instruct_name}_{speed}.wav"

                print(f"🎧 Generating: {filename}")

                for i, j in enumerate(cosyvoice.inference_instruct2(
                    text,
                    instruct,
                    wav_path,
                    zero_shot_spk_id=spk_id,
                    speed=speed,
                    stream=False
                )):
                    torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)

    print("✅ 모든 실험 완료")

if __name__ == '__main__':
    run_experiments()