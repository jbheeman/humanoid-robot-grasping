# Local voice over ROS 2

`scripts/gb10/voice_ros_bridge.py` is an opt-in, GB10-local voice bridge. It does not publish arm, locomotion, or Unitree commands:

```text
microphone -> faster-whisper -> /g1/voice/transcript -> Ollama -> /g1/voice/reply
```

The same node can skip audio and accept a text request from `/g1/voice/input`, so the ROS/Ollama path is testable before connecting a microphone.

## Why this split

Ollama's documented chat endpoint accepts chat messages and its documented multimodal attachment is an `images` array; audio is not a general documented chat attachment. Keep speech recognition as a small local ASR stage and use the existing fast Ollama text model for reasoning. This keeps audio local to GB10 and lets YOLO and ASR share its GPU.

An audio-capable “omni” model would still perform ASR internally and its audio prompt format/runtime support is model-specific. It can be benchmarked later, but is not the lowest-risk first integration.

## Install and run on GB10

Install the three optional local-audio packages in the GB10 project environment:

```bash
uv pip install faster-whisper sounddevice soundfile
ollama list
source scripts/shared/ros-env.sh
g1_source_ros "$PWD" "${GB10_ROS_DISTRO:-jazzy}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/gb10/voice_ros_bridge.py --model <model-from-ollama-list> --microphone
```

Press Enter, speak for five seconds, then wait for the reply. The bridge uses reliable `std_msgs/String` topics:

- `/g1/voice/input` — text request (subscription)
- `/g1/voice/transcript` — recognized/request text
- `/g1/voice/reply` — Ollama response
- `/g1/voice/status` — `ready`, `thinking`, or error state

Test without audio:

```bash
python scripts/gb10/voice_ros_bridge.py --model <model>
ros2 topic pub --once /g1/voice/input std_msgs/msg/String "{data: 'What can you see?'}"
ros2 topic echo /g1/voice/reply --once
```

## Optional local spoken reply

Install a local Piper binary and local voice model, then pass its command with a `{wav}` output placeholder. Reply text is sent to standard input only, never interpolated into the command:

```bash
python scripts/gb10/voice_ros_bridge.py --model <model> --microphone \
  --tts-command 'piper --model /opt/piper/en_US-lessac-medium.onnx --output_file {wav}'
```
