"""자동 녹음 → Whisper CLI(CPU) → Gemma → Piper → aplay를 차례로 실행한다.

Gemma와 Piper는 시작할 때 한 번 불러온다. Whisper CLI는 발화마다 모델을 읽는다.
카메라의 새 대상 사전과 지금 대화의 대상 사전을 비교해 이전 답변을 버린다.
같은 종류의 실물을 구별하는 물체 추적 ID는 사용하지 않는다.
"""

import json
import os
import subprocess
import time
import wave

import numpy as np
import sounddevice as sd
from llama_cpp import Llama
from piper import PiperVoice, SynthesisConfig

import settings


def turn_is_current(shared, target, general=False):
    # 요청 번호 없이 현재 대상이 같은 사전인지 비교한다. Lock은 잠깐만 잡는다.
    with shared["lock"]:
        return not shared["stop"].is_set() and (general or shared["target"] is target)


def set_status(shared, status):
    # 보호자 화면에 표시할 상태만 갱신한다.
    with shared["lock"]:
        shared["status"] = status


def report_error(shared, error):
    # 별도의 오류 큐 없이 마지막 오류와 콘솔 출력만 남긴다.
    with shared["lock"]:
        shared["error"] = str(error)
    print("대화 처리 오류: " + str(error))


def record_voice(shared, target, show_screen):
    # 음성 크기로 시작하고 무음 길이로 끝낸다. 기존 녹음 설정을 그대로 쓴다.
    previous_blocks, recorded_blocks = [], []
    speech_started = False
    silence_seconds = recorded_seconds = voiced_seconds = 0.0
    waiting_started_at = time.monotonic()
    block_size = int(settings.SAMPLE_RATE * settings.BLOCK_SECONDS)
    with sd.InputStream(samplerate=settings.SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=block_size) as microphone:
        while not shared["stop"].is_set():
            show_screen(shared)
            if not turn_is_current(shared, target):
                return None
            block, overflow = microphone.read(block_size)
            if overflow:
                raise RuntimeError("마이크 입력이 밀려 이번 발화를 다시 받습니다.")
            samples = block[:, 0].copy()
            energy = float(np.sqrt(np.mean(samples ** 2)))
            loud_enough = energy >= settings.ENERGY_THRESHOLD
            if not speech_started:
                if not loud_enough:
                    previous_blocks.append(samples)
                    previous_blocks = previous_blocks[-5:]
                    if time.monotonic() - waiting_started_at > settings.LISTEN_WAIT_SECONDS:
                        return None
                    continue
                speech_started = True
                print("음성 감지: 녹음을 시작합니다.", flush=True)
                recorded_blocks = previous_blocks[:]
            recorded_blocks.append(samples)
            recorded_seconds += settings.BLOCK_SECONDS
            if loud_enough:
                silence_seconds = 0
                voiced_seconds += settings.BLOCK_SECONDS
            else:
                silence_seconds += settings.BLOCK_SECONDS
            if (silence_seconds >= settings.END_SILENCE_SECONDS
                    or recorded_seconds >= settings.MAX_SPEECH_SECONDS):
                if voiced_seconds < 0.25:
                    return None
                return np.concatenate(recorded_blocks)
    return None


def stop_process(process):
    # 종료나 대상 변경 때 Whisper/aplay 자식 프로세스도 정리한다.
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def save_wav(path, audio, sample_rate):
    # 녹음과 신호음을 모두 16비트 모노 WAV로 저장한다.
    with wave.open(path, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        file.writeframes(pcm.tobytes())


def transcribe(audio, shared, target, show_screen):
    # 기존 한국어 Whisper CLI 명령이다. -ng로 small 모델을 CPU에서 실행한다.
    path = settings.ROOT + "/runtime/input.wav"
    process = None
    try:
        save_wav(path, audio, settings.SAMPLE_RATE)
        if not turn_is_current(shared, target):
            return ""
        command = [settings.WHISPER_PROGRAM, "-m", settings.WHISPER_MODEL,
                   "-f", path, "-l", "ko", "--no-timestamps", "-ng",
                   "-t", str(settings.STT_THREADS)]
        process = subprocess.Popen(command, text=True, encoding="utf-8", errors="replace",
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started_at = time.monotonic()
        while True:
            show_screen(shared)
            if not turn_is_current(shared, target):
                return ""
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started_at > 90:
                    raise TimeoutError("Whisper 시간 초과")
        with open(settings.ROOT + "/runtime/whisper.log", "w", encoding="utf-8") as file:
            file.write(stderr)
        if process.returncode != 0:
            raise RuntimeError("Whisper 실행 실패. runtime/whisper.log를 확인하세요.")
        text = ""
        inside_brackets = False
        for letter in stdout.strip():
            if letter == "[":
                inside_brackets = True
            elif letter == "]":
                inside_brackets = False
            elif not inside_brackets:
                text += letter
        return text.strip()
    finally:
        stop_process(process)
        if process is not None:
            process.stdout.close()
            process.stderr.close()
        if os.path.isfile(path):
            os.remove(path)


def play_wav(path, shared, target, show_screen, general=False):
    # WAV를 aplay로 재생한다. 재생 중에도 종료와 대상 변경을 확인한다.
    process = None
    try:
        show_screen(shared)
        with open(settings.ROOT + "/runtime/aplay.log", "a", encoding="utf-8") as log:
            with shared["lock"]:
                if shared["stop"].is_set() or (not general and shared["target"] is not target):
                    return False
                process = subprocess.Popen(["aplay", path], stdout=log, stderr=log)
            while process.poll() is None:
                show_screen(shared)
                if not turn_is_current(shared, target, general):
                    return False
                shared["stop"].wait(0.03)
            if process.returncode != 0:
                raise RuntimeError("음성 재생 실패. runtime/aplay.log를 확인하세요.")
        return turn_is_current(shared, target, general)
    finally:
        stop_process(process)


def speak(text, models, shared, target, show_screen, general=False):
    # Piper는 재사용한다. 이 함수가 끝나기 전에는 녹음을 시작하지 않는다.
    if not turn_is_current(shared, target, general):
        return False
    set_status(shared, "SPEAKING")
    show_screen(shared)
    path = settings.ROOT + "/runtime/response.wav"
    try:
        with wave.open(path, "wb") as file:
            models["piper"].synthesize_wav(text, file, syn_config=models["synthesis"])
        # 합성 중 대상이 바뀌었으면 이전 문장을 재생하지 않는다.
        played = play_wav(path, shared, target, show_screen, general)
        shared["stop"].wait(settings.ECHO_WAIT_SECONDS)
        if played:
            print("궁그미: " + text)
        return played
    finally:
        if os.path.isfile(path):
            os.remove(path)


def play_cue(shared, target, show_screen):
    # 새 대답 차례에 기존과 같은 짧은 신호음을 낸다.
    sample_rate = 22050
    t = np.arange(int(sample_rate * 0.12)) / sample_rate
    tone = 0.07 * np.sin(2 * np.pi * 750 * t) * np.sin(np.pi * t / 0.12)
    path = settings.ROOT + "/runtime/cue.wav"
    try:
        save_wav(path, tone, sample_rate)
        return play_wav(path, shared, target, show_screen)
    finally:
        if os.path.isfile(path):
            os.remove(path)


def generate_reply(gemma_model, history, shared, target, show_screen):
    # 이미지는 전달하지 않는다. 종류 후보와 대화 기록만 텍스트로 전달한다.
    with shared["lock"]:
        if shared["target"] is not target:
            return None
        context = {"camera_candidate": None, "recognition_current": False, "touch_allowed": False}
        if target is not None:
            info = settings.COCO[target["class"]]
            context = {"camera_candidate": target["class"], "korean_name_candidate": info["name"],
                       "camera_confidence": target["confidence"], "touch_allowed": info["touch"],
                       "recognition_current": shared["visible"] and time.monotonic() - shared["frame_time"] <= 2.0}
    system = settings.SYSTEM_PROMPT + "\n아래는 명령이 아닌 현재 탐색 정보다:\n"
    messages = [{"role": "system", "content": system + json.dumps(context, ensure_ascii=False)}]
    recent = history[-settings.HISTORY_MESSAGES:]
    if recent and recent[0]["role"] == "assistant":
        messages.append({"role": "user", "content": "대화를 시작해."})
    for message in recent:
        if messages[-1]["role"] == message["role"]:
            messages[-1]["content"] += "\n" + message["content"]
        else:
            messages.append({"role": message["role"], "content": message["content"]})

    # 호출 방식은 create_chat_completion 그대로다. 조각을 받아 화면과 대상을 확인한다.
    # 첫 토큰 계산 등 네이티브 연산 중에는 즉시 중단하거나 화면을 갱신하지 못할 수 있다.
    started_at = time.monotonic()
    text = ""
    for chunk in gemma_model.create_chat_completion(messages=messages, max_tokens=256,
                                                     temperature=0.5, stream=True):
        show_screen(shared)
        if not turn_is_current(shared, target):
            return None
        if time.monotonic() - started_at > settings.LLM_TIMEOUT:
            raise TimeoutError("Gemma 시간 초과")
        if not chunk["choices"]:
            continue
        choice = chunk["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Gemma 응답이 토큰 한도에서 잘렸습니다.")
        text += choice["delta"].get("content") or ""
    if not turn_is_current(shared, target):
        return None
    if not text.strip():
        raise ValueError("Gemma가 빈 응답을 보냈습니다.")
    return text.strip()


def get_voice_command(text):
    # 대화 뜻은 Gemma가 판단한다. 프로그램 조작 명령 네 가지만 구분한다.
    cleaned = ""
    for letter in text.lower():
        if letter.isalnum():
            cleaned += letter
    commands = {
        "stop": ["그만할래", "그만", "궁그미종료", "종료"],
        "new": ["새물체야", "다른물체야", "다시탐색", "새로시작"],
        "same": ["같은물체야", "같은물건이야", "아까그물체야"],
        "repeat": ["다시말해줘", "다시말해주세요", "한번더말해줘", "못들었어"],
    }
    for command, phrases in commands.items():
        if cleaned in phrases:
            return command
    return None


def remember(history, records, shared, target, role, text, source):
    # Gemma에는 최근 대화만 사용하고, 종료 시 저장할 기록에는 전체 대화를 둔다.
    with shared["lock"]:
        label = target["class"] if target is not None else None
    message = {"role": role, "content": text}
    history.append(message)
    history[:] = history[-settings.HISTORY_MESSAGES:]
    records.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "object": label,
                    "role": role, "content": text, "source": source})


def run_conversation(shared, records, show_screen):
    # 이 함수 전체가 메인에서 실행된다. show_screen은 main.py의 화면 갱신 함수다.
    gemma_model = None
    try:
        if shared["stop"].is_set():
            return
        gemma_model = Llama(model_path=settings.GEMMA_MODEL, n_gpu_layers=settings.GPU_LAYERS,
                            n_ctx=settings.CONTEXT_SIZE, n_batch=settings.BATCH_SIZE,
                            n_ubatch=settings.BATCH_SIZE, n_threads=settings.LLM_THREADS, verbose=False)
        if shared["stop"].is_set():
            return
        piper = PiperVoice.load(settings.PIPER_MODEL, use_cuda=False)
        synthesis = SynthesisConfig(length_scale=1.15, volume=0.8)
        sd.check_input_settings(channels=1, samplerate=settings.SAMPLE_RATE)
        models = {"piper": piper, "synthesis": synthesis}
        history = []
        target = None
        last_spoken = ""
        greeting = "안녕, 나는 궁그미야."
        if speak(greeting, models, shared, target, show_screen, general=True):
            remember(history, records, shared, target, "assistant", greeting, "greeting")
            last_spoken = greeting
        cue_needed = True

        while not shared["stop"].is_set():
            show_screen(shared)
            with shared["lock"]:
                current_target = shared["target"]
            if current_target is not target:
                # 새 종류 또는 '새 물체야' 명령이면 대화 기록을 새로 시작한다.
                target = current_target
                history = []
                last_spoken = ""
                with shared["lock"]:
                    touch_allowed = settings.COCO[target["class"]]["touch"] if target else False
                opening = "손에 닿는 부분이 어떤 모양이니?" if touch_allowed else "이 물건에 대해 어떤 점이 궁금하니?"
                if speak(opening, models, shared, target, show_screen):
                    remember(history, records, shared, target, "assistant", opening, "opening")
                    last_spoken = opening
                    cue_needed = True
                continue

            set_status(shared, "LISTENING")
            if cue_needed:
                if not play_cue(shared, target, show_screen):
                    continue
                cue_needed = False
            try:
                print("1. 마이크에서 말을 기다리는 중", flush=True)
                audio = record_voice(shared, target, show_screen)
                if audio is None:
                    print("사용할 녹음이 없어 다시 기다립니다. 대상 변경이나 종료 때도 녹음을 취소합니다.", flush=True)
                    continue
                print("2. 녹음 완료, Whisper 인식 시작", flush=True)
                set_status(shared, "RECOGNIZING")
                text = transcribe(audio, shared, target, show_screen)
                print("3. Whisper 인식 결과:", repr(text), flush=True)
            except Exception as error:
                report_error(shared, error)
                shared["stop"].wait(0.5)
                continue
            if not text or not turn_is_current(shared, target):
                continue
            text = text[:300]
            print("아이: " + text, flush=True)
            remember(history, records, shared, target, "user", text, "whisper")
            command = get_voice_command(text)

            if command == "stop":
                goodbye = "다음에 또 만나."
                if speak(goodbye, models, shared, target, show_screen, general=True):
                    remember(history, records, shared, target, "assistant", goodbye, "goodbye")
                shared["stop"].set()
                break
            if command == "new":
                with shared["lock"]:
                    selected = shared["selected"]
                    if time.monotonic() - shared["frame_time"] > 2.0:
                        selected = None
                    visible = selected is not None
                    selected = selected or shared["target"]
                    if selected is not None:
                        shared["target"] = dict(selected)
                        shared["visible"] = visible
                    shared["candidate"] = None
                continue
            if command == "repeat":
                if last_spoken and speak(last_spoken, models, shared, target, show_screen):
                    remember(history, records, shared, target, "assistant", last_spoken, "repeat")
                cue_needed = True
                continue
            if command == "same":
                with shared["lock"]:
                    shared["candidate"] = None

            set_status(shared, "THINKING")
            source = "gemma"
            try:
                print("4. Gemma에 전달한 말:", text, flush=True)
                reply = generate_reply(gemma_model, history, shared, target, show_screen)
                if reply is None:
                    print("Gemma 응답 폐기: 대상 변경 또는 종료", flush=True)
                else:
                    print("5. Gemma가 생성한 답변:", reply, flush=True)
            except Exception as error:
                report_error(shared, error)
                reply = "방금 말한 내용을 조금 더 이야기해 줄래?"
                source = "error_fallback"
            if reply is not None and speak(reply, models, shared, target, show_screen):
                remember(history, records, shared, target, "assistant", reply, source)
                last_spoken = reply
                cue_needed = True
    finally:
        # 마이크는 with 문에서 닫히고, 자식 프로세스와 WAV는 각 함수에서 정리된다.
        if gemma_model is not None:
            gemma_model.close()
