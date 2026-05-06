import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import os

def run_experiments():
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    wav_path = './asset/myvoice_2.wav'
    spk_id = 'my_korean_spk'

    os.makedirs("outputs", exist_ok=True)

    texts = {
        "normal": (
            "안녕하세요. 오늘 날씨가 참 좋네요. 이렇게 좋은 날에는 산책하기 딱 좋은 것 같아요. "
            "저는 요즘 독서를 많이 하고 있는데, 여러분은 어떤 취미를 즐기시나요?<|endofprompt|>"
        ),
        "phoneme": (
            "안 녕 하 세 요. 오 늘 날 씨 가 참 좋 네 요. 이 렇 게 좋 은 날 에 는 산 책 하 기 딱 좋 은 것 같 아 요. "
            "저 는 요 즘 독 서 를 많 이 하 고 있 는 데, 여 러 분 은 어 떤 취 미 를 즐 기 시 나 요.<|endofprompt|>"
        ),
        "mixed": (
            "안녕하세요. 오늘 날씨가 참 좋네요. "
            "산책하기 딱 좋은 것 같아요. "
            "저는 요즘 독서를 많이 하고 있는데, "
            "여러분은 어떤 취미를 즐기시나요?<|endofprompt|>"
        )
    }

    instructs = {
        "neutral": "You are a helpful assistant.<|endofprompt|>",
        "friendly": "You are a helpful assistant. Speak in a friendly tone.<|endofprompt|>",
    }

    speeds = [1.0, 1.2]

    # 스피커 등록
    cosyvoice.add_zero_shot_spk(texts["normal"], wav_path, spk_id)

    for t_name, text in texts.items():
        for i_name, instruct in instructs.items():
            for speed in speeds:

                filename = f"outputs/{t_name}_{i_name}_{speed}.wav"

                result = None
                for j in cosyvoice.inference_instruct2(
                    text,
                    instruct,
                    wav_path,
                    zero_shot_spk_id=spk_id,
                    speed=speed,
                    stream=False
                ):
                    result = j

                if result is not None:
                    torchaudio.save(filename, result['tts_speech'], cosyvoice.sample_rate)
                    print("✅", filename)
                else:
                    print("❌ failed:", filename)

if __name__ == '__main__':
    run_experiments()