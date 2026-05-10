import sys
import os
import re
import torch
import torchaudio

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel


# -------------------------
# util
# -------------------------
def preprocess_line(line):
    line = re.sub(r'\[pause\]', ',', line)
    line = re.sub(r'\[(?!laughter|breath)[^\]]+\]', '', line)
    line = re.sub(r' +', ' ', line).strip()
    return line


def make_silence(sample_rate, duration_sec=0.4):
    return torch.zeros(1, int(sample_rate * duration_sec))


def infer_line(cosyvoice, text, instruct, prompt_wav, speed=1.0):
    """
    zero_shot_spk_id를 사용하지 않고 prompt_wav만 직접 넘김
    (spk_id 경로를 쓰면 text_prompt가 LLM 입력에 concat되어 혼입 발생)
    """
    audio = None
    for out in cosyvoice.inference_instruct2(
        text,
        instruct,
        prompt_wav,
        stream=False,
        speed=speed
    ):
        audio = out["tts_speech"].cpu()
    return audio


# -------------------------
# main
# -------------------------
def run():

    cosyvoice = AutoModel(
        model_dir='pretrained_models/Fun-CosyVoice3-0.5B'
    )

    os.makedirs("outputs_0507_fixed", exist_ok=True)

    prompt_wav = "./asset/final_prompt.wav"
    
    instruct = (
        "You are a helpful assistant. "
        "Please preserve the original speaker voice as much as possible. "
        "Speak in a very soft, breathy Korean tone — gentle and unhurried, like a quiet murmur to oneself. "
        "The mood is calm and slightly weary, not mocking — just softly observing. "
        "Any laughter should be a tired, hollow exhale not a chuckle, just a soft breath of amusement. "
        "<|endofprompt|>"
    )

    scene1_lines = [
        "당신을 믿으려면 [breath] 당신이 누군지 알아야 한대.",
        "근데, [breath] 어제 한 여자를 만났대.",
        "그 여자 이름이 [breath] 차 은상이래.",
        "근데 차은상한테 궁금한 게 생겼대.",
        "혹시, [breath] <strong>나 너 좋아하냐?</strong>"
    ]
    
    scene_2_lines = [
        "난 그냥 니가 가서 쓸쓸했고, [breath]",
        "돌아와서 좋고, [breath]",
        "니 비밀은 무겁고 [breath] 그냥 그래.",
        "내가 뭘 어떻게 한대?"]
    
    scene_3_lines = [
        "넌 처음부터 나한테 여자였고, [breath]",
        "지금도 여자야.",
        "앞으로는, [breath] 내 첫사랑이고.",
        "마주치면 인사하지 말자.",
        "잘 지내냐, 안부도 묻지 말자.",
        "시간이 아주 오래 지나도, 그땐 그랬지... 하면서 추억인 척,",
        "웃으며 아는 척 하지도 말자."
    ]
    
    scene_4_lines = [
        "어떻게 하면 될지 알려줘? [breath]",
        "너 내일 당장 우리 집에서 나가.",
        "못 나가?",
        "학교도 계속 다니고 싶어?",
        "그럼 지금부터 나 좋아해, [breath]",
        "[strong] 가능하면 진심으로.",
        "난 니가 좋아졌어."
    ]
    
    scene_5_lines = [
        "야..!",
        "넌 왜 맨날 이런데서 자냐..",
        "지켜주고 싶게."
    ]
    
    scene_6_lines = [
        "[laughter] 사탄들의 학교에,",
        "루시퍼의 등장이라... [breath]",
        "재밌어지겠네."
    ]
    
    scenes = [
        # scene1_lines,
        # scene_2_lines,
        # scene_3_lines,
        # scene_4_lines,
        # scene_5_lines,
        scene_6_lines,
    ]

    speeds = [1.0]

    for speed in speeds:
        for scene_idx, scene_lines in enumerate(scenes, start=1):
            print(f"\n🎬 scene{scene_idx} | speed={speed}")

            parts = []

            for idx, line in enumerate(scene_lines):
                processed = preprocess_line(line)
                print(f"  [{idx+1}/{len(scene_lines)}] {processed}")

                audio = infer_line(cosyvoice, processed, instruct, prompt_wav, speed)

                if audio is not None:
                    parts.append(audio)
                else:
                    print(f"  ⚠️ line {idx+1} returned None, skipping")

            if not parts:
                print(f"⚠️ scene{scene_idx} no audio generated, skipping save")
                continue

            # 대사 사이에 짧은 silence 삽입해서 이어붙이기
            silence = make_silence(cosyvoice.sample_rate, duration_sec=0.35)
            combined = parts[0]
            for part in parts[1:]:
                combined = torch.cat([combined, silence, part], dim=1)

            out_path = f"outputs_0507/scene6_normal_{speed}.wav"
            torchaudio.save(out_path, combined, cosyvoice.sample_rate)
            print(f"✅ saved: {out_path}")

if __name__ == "__main__":
    run()