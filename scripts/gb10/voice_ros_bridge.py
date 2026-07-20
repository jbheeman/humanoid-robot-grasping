#!/usr/bin/env python3
"""Local push-to-talk voice bridge for the GB10.

This node deliberately has no robot-control publisher. It turns local
microphone audio (or a ROS text message) into text, sends it to the local
Ollama server, and publishes the transcript and reply as ROS String messages.
"""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

INPUT_TOPIC = "/g1/voice/input"
TRANSCRIPT_TOPIC = "/g1/voice/transcript"
REPLY_TOPIC = "/g1/voice/reply"
STATUS_TOPIC = "/g1/voice/status"
DEFAULT_SYSTEM_PROMPT = (
    "You are the G1's local voice assistant. Be concise and helpful. You are conversational only: "
    "never claim to move the robot or execute physical actions. Movement requires the separate guarded ROS workflow."
)


def ollama_chat(endpoint: str, model: str, system_prompt: str, text: str, timeout: float) -> str:
    """Return one non-streaming reply using Ollama's local /api/chat API."""
    payload = json.dumps({"model": model, "stream": False, "keep_alive": "15m", "messages": [
        {"role": "system", "content": system_prompt}, {"role": "user", "content": text},
    ]}).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    reply = result.get("message", {}).get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError(f"Ollama returned no chat content: {result!r}")
    return reply.strip()


def transcribe_wav(wav_path: Path, model_size: str, device: str, compute_type: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Microphone mode needs faster-whisper. Run: uv pip install faster-whisper") from exc
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(wav_path), beam_size=1, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def record_wav(wav_path: Path, seconds: float, sample_rate: int, device: str | None) -> None:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Microphone mode needs sounddevice and soundfile. Run: uv pip install sounddevice soundfile") from exc
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1,
                       dtype="int16", device=device)
    sd.wait()
    sf.write(wav_path, recording, sample_rate, subtype="PCM_16")


def speak(text: str, command_template: str) -> None:
    """Pipe text into local TTS. Template must contain {wav} output path."""
    if "{wav}" not in command_template:
        raise RuntimeError("--tts-command must include {wav} for its output WAV path")
    with tempfile.TemporaryDirectory(prefix="g1-voice-") as directory:
        wav_path = Path(directory) / "reply.wav"
        command = [part.replace("{wav}", str(wav_path)) for part in shlex.split(command_template)]
        subprocess.run(command, input=text, text=True, check=True, timeout=60)
        subprocess.run(["aplay", "-q", str(wav_path)], check=True, timeout=60)


class VoiceBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.args, self.rclpy, self.String = args, rclpy, String
        self.requests: queue.Queue[str | None] = queue.Queue()
        self.node = rclpy.create_node("g1_local_voice")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.transcript_pub = self.node.create_publisher(String, TRANSCRIPT_TOPIC, qos)
        self.reply_pub = self.node.create_publisher(String, REPLY_TOPIC, qos)
        self.status_pub = self.node.create_publisher(String, STATUS_TOPIC, qos)
        self.node.create_subscription(String, INPUT_TOPIC, self._on_input, qos)
        self.worker = threading.Thread(target=self._run_requests, daemon=True)

    def _publish(self, publisher: object, text: str) -> None:
        message = self.String(); message.data = text; publisher.publish(message)

    def status(self, text: str) -> None:
        self.node.get_logger().info(text); self._publish(self.status_pub, text)

    def _on_input(self, message: object) -> None:
        text = str(message.data).strip()
        if text: self.requests.put(text)

    def _run_requests(self) -> None:
        while True:
            text = self.requests.get()
            if text is None: return
            self._publish(self.transcript_pub, text); self.status("thinking")
            try:
                reply = ollama_chat(self.args.ollama_url, self.args.model, self.args.system_prompt, text, self.args.timeout)
                self._publish(self.reply_pub, reply); self.status("speaking" if self.args.tts_command else "ready")
                if self.args.tts_command: speak(reply, self.args.tts_command); self.status("ready")
            except Exception as exc:
                self.status(f"error: {exc}")

    def microphone_loop(self) -> None:
        self.status("microphone ready; press Enter to record, or Ctrl-C to exit")
        while self.rclpy.ok():
            try: input()
            except EOFError: return
            with tempfile.TemporaryDirectory(prefix="g1-voice-") as directory:
                wav_path = Path(directory) / "request.wav"; self.status(f"recording for {self.args.record_seconds:g} seconds")
                try:
                    record_wav(wav_path, self.args.record_seconds, self.args.sample_rate, self.args.audio_device)
                    self.status("transcribing")
                    text = transcribe_wav(wav_path, self.args.whisper_model, self.args.whisper_device, self.args.compute_type)
                except Exception as exc:
                    self.status(f"error: {exc}"); continue
            if text: self.requests.put(text)
            else: self.status("no speech recognized")

    def run(self) -> None:
        self.worker.start(); self.status("ready")
        if self.args.microphone: self.microphone_loop()
        else: self.rclpy.spin(self.node)

    def close(self) -> None:
        self.requests.put(None); self.worker.join(timeout=2); self.node.destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local microphone/text -> Ollama -> ROS voice bridge")
    parser.add_argument("--model", required=True, help="Installed Ollama text model, e.g. qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT); parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--microphone", action="store_true"); parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000); parser.add_argument("--audio-device")
    parser.add_argument("--whisper-model", default="small.en"); parser.add_argument("--whisper-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--tts-command", help="Optional local TTS command: must include {wav}; reply text goes to stdin.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.record_seconds <= 0 or args.sample_rate <= 0 or args.timeout <= 0: raise SystemExit("record seconds, sample rate, and timeout must be positive")
    import rclpy
    rclpy.init(); bridge = VoiceBridge(args)
    try: bridge.run()
    except KeyboardInterrupt: pass
    finally: bridge.close(); rclpy.shutdown()


if __name__ == "__main__": main()
