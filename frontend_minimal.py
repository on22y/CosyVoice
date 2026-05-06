import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

def main():
    # CosyVoice SFT 모델 로드 (한국어/중국어/영어 등)
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')

    # Windows용 최소 실행: frontend 비활성화
    sample_texts = [
        "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        "And then later on, fully acquiring that company."
    ]

    speakers = ["中文女", "中文男", "英文女"]

    for idx, (txt, spk) in enumerate(zip(sample_texts, speakers)):
        print(f"生成 음성 {idx}... [{spk}]")
        for i, j in enumerate(
            cosyvoice.inference_sft(txt, spk, stream=False, text_frontend=False)  # frontend 강제 비활성화
        ):
            filename = f"minimal_sft_{idx}_{i}.wav"
            torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
            print(f"Saved: {filename}")

    print("모든 음성 생성 완료!")

if __name__ == "__main__":
    main()