import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import torch
import os

# 문장 사이 pause
def add_pause(sample_rate, duration=0.4):
    return torch.zeros(1, int(sample_rate * duration))

# 안전한 trim (energy 기반)
def trim_silence(audio, threshold=0.01):
    """
    무음 구간 제거 (앞/뒤)
    """
    energy = audio.abs()
    mask = energy > threshold

    if mask.sum() == 0:
        return audio

    start = mask.nonzero()[0][1]
    end = mask.nonzero()[-1][1]

    return audio[:, start:end]

# 생성
def generate_scene(cosyvoice, wav_path, spk_id, lines, speed=1.0):
    """
    문장 단위로 생성 후 이어붙이기
    """
    outputs = []
    pause = add_pause(cosyvoice.sample_rate, 0.45)

    for idx, line in enumerate(lines):
        text = line.strip() + "<|endofprompt|>"

        print(f"👉 Generating line {idx+1}: {line}")

        result_audio = None

        for j in cosyvoice.inference_instruct2(
            text,
            "You are a Korean speaker. Match the speaker's voice as closely as possible. Maintain natural pacing, realistic pauses, and subtle emotion. Do not exaggerate expression. Do not add or repeat words.<|endofprompt|>",
            wav_path,
            zero_shot_spk_id=spk_id,
            speed=speed,
            stream=False
        ):
            result_audio = j['tts_speech']

        # 앞/뒤 무음 구간 제거
        result_audio = trim_silence(result_audio)

        outputs.append(result_audio)
        outputs.append(pause)

    # 이어붙이기
    final_audio = torch.cat(outputs, dim=1)
    return final_audio


def run():
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    # wav_path = './asset/myvoice_2.wav'   # prompt wav
    # spk_id = 'my_korean_spk'

    os.makedirs("outputs_0506", exist_ok=True)
    
    # prompt wav
    prompt_wavs = {
        "normal": "./asset/spk_normal.wav",

        "scene1": "./asset/spk_scene1.wav",
        # "scene2": "./asset/spk_scene2.wav",
        # "scene3": "./asset/spk_scene3.wav",
        # "scene4": "./asset/spk_scene4.wav",
        # "scene5": "./asset/spk_scene5.wav",
        # "scene6": "./asset/spk_scene6.wav",
    }

    # Zero-shot speaker 등록 (한 번만)
    # cosyvoice.add_zero_shot_spk(
    #     "안녕하세요. 테스트 문장입니다.<|endofprompt|>",
    #     wav_path,
    #     spk_id
    # )
    # cosyvoice.save_spkinfo()
    
    # spk_id 분리
    spk_id_map = {
        "normal": "spk_normal",

        "scene1": "spk_scene1",
        # "scene2": "spk_scene2",
        # "scene3": "spk_scene3",
        # "scene4": "spk_scene4",
        # "scene5": "spk_scene5",
        # "scene6": "spk_scene6",
    }
    
    # zero-shot 등록
    for key, wav_path in prompt_wavs.items():
        cosyvoice.add_zero_shot_spk(
            "<|endofprompt|>",
            wav_path,
            spk_id_map[key]
        )

    cosyvoice.save_spkinfo()

    # 명장면 예시
    
    scene_1 = [
      "당신을 믿으려면 [breath] 당신이 누군지 알아야 한대.",
      "근데, [breath] 어제 한 여자를 만났대.",
      "그 여자 이름이 [breath] 차 은상이래.",
      "근데 차은상한테 궁금한 게 생겼대.",
      "혹시, [breath] 나 너 좋아하냐?"
    ]

    scene_2 = [
        "난 그냥 니가 가서 쓸쓸했고, [breath]",
        "돌아와서 좋고, [breath]",
        "니 비밀은 무겁고 [breath] 그냥 그래.",
        "내가 뭘 어떻게 한대?"
    ]

    scene_3 = [
        "넌 처음부터 나한테 여자였고, [breath]",
        "지금도 여자야.",
        "앞으로는, [breath] 내 첫사랑이고.",
        "마주치면 인사하지 말자.",
        "잘 지내냐, 안부도 묻지 말자.",
        "시간이 아주 오래 지나도, 그땐 그랬지... 하면서 추억인 척,",
        "웃으며 아는 척 하지도 말자."
    ]

    scene_4 = [
        "어떻게 하면 될지 알려줘? [breath]",
        "너 내일 당장 우리 집에서 나가.",
        "못 나가?",
        "학교도 계속 다니고 싶어?",
        "그럼 지금부터 나 좋아해, [breath]",
        "[strong] 가능하면 진심으로.",
        "난 니가 좋아졌어."
    ]
    
    scene_5 = [
      "야!",
      "넌 왜 맨날 이런데서 자냐..",
      "지켜주고 싶게."
    ]
    
    scene_6 = [
      "[laughter] 사탄들의 학교에,",
      "루시퍼의 등장이라... [breath]",
      "재밌어지겠네."
    ]

    scenes = {
    "scene1": scene_1,
    # "scene2": scene_2,
    # "scene3": scene_3,
    # "scene4": scene_4,
    # "scene5": scene_5,
    # "scene6": scene_6,
}

    # speeds = [0.9, 1.0, 1.2]
    speeds = [1.0]

    # 실행

    for scene_name, lines in scenes.items():
        for speed in speeds:

            # normal prompt
            print(f"\n🎬 {scene_name} | normal prompt")

            audio = generate_scene(
                cosyvoice,
                prompt_wavs["normal"],
                spk_id_map["normal"],
                lines,
                speed=speed
            )

            torchaudio.save(
                f"outputs_0506/{scene_name}_normal_{speed}.wav",
                audio,
                cosyvoice.sample_rate
            )

            # scene-specific prompt
            print(f"\n🎬 {scene_name} | scene prompt")

            audio = generate_scene(
                cosyvoice,
                prompt_wavs[scene_name],
                spk_id_map[scene_name],
                lines,
                speed=speed
            )

            torchaudio.save(
                f"outputs_0506/{scene_name}_matched_{speed}.wav",
                audio,
                cosyvoice.sample_rate
            )


if __name__ == "__main__":
    run()