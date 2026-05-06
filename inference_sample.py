import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def main():
    # 모델 로드
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    # 한국어 텍스트
    text = "Hello, this is a test of text to speech."

    # 프롬프트 (목소리 wav 파일)
    prompt_wav = "./asset/myvoice_1.wav"

    # instruct prompt (스타일 + 한국어)
    instruct = "You are a helpful assistant.<|endofprompt|>Speak in a calm and natural voice."

    # inference
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            text,
            instruct,
            prompt_wav,
            stream=False
        )
    ):
        torchaudio.save(f'output_1440.wav', j['tts_speech'], cosyvoice.sample_rate)


if __name__ == "__main__":
    main()
