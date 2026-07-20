"""Ollama-like terminal for Unitree UnifoLM-VLA on the GB10."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch
from typing import Any, Sequence

import numpy as np
from PIL import Image

from object_tracking.arm_tracking.joints import BODY_JOINT_NAMES
from object_tracking.arm_tracking.calibration import load_calibration
from object_tracking.arm_tracking.geometry import Plane, SupportRegion
from object_tracking.arm_tracking.ik_solver import G1RightArmIK, default_urdf_path
from object_tracking.ros2_tracking import RosTrackingTransport
from object_tracking.unifolm_vla import compose_pose23, parse_action_chunk


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[38;5;114m"
AMBER = "\033[38;5;179m"
RED = "\033[38;5;203m"
CYAN = "\033[38;5;110m"
MAX_OBSERVATION_AGE_MS = 500.0
EXECUTION_EFFECTORS = ("right hand", "right arm", "right palm")
EXECUTION_VERBS = (
    "approach",
    "extend",
    "hold",
    "lift",
    "lower",
    "move",
    "raise",
    "reach",
    "retract",
    "touch",
)
GREETING_ONLY = {"hello", "hey", "hi", "sup", "yo"}
BNB_QUANTIZATION_STATE_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
)


class VLAError(RuntimeError):
    pass


def _load_checkpoint_state(model: Any, state: dict[str, Any], *, quantized: bool) -> None:
    """Load a full checkpoint while validating bitsandbytes bookkeeping.

    PyTorch does not register the serialized NF4 quantization metadata as
    model buffers, so vanilla ``load_state_dict(strict=True)`` rejects those
    keys even when every actual model weight matches.  In the quantized case
    only, accept that exact metadata family and continue to fail closed for
    missing weights or any other unexpected key.
    """

    if not quantized:
        model.load_state_dict(state, strict=True)
        return
    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(
        key
        for key in incompatible.unexpected_keys
        if not key.endswith(BNB_QUANTIZATION_STATE_SUFFIXES)
    )
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing[:8])}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(unexpected[:8])}")
        raise VLAError("quantized VLA checkpoint does not match the model (" + "; ".join(details) + ")")


def _color(value: str, code: str, enabled: bool) -> str:
    return f"{code}{value}{RESET}" if enabled else value


def _is_explicit_right_arm_instruction(instruction: str) -> bool:
    normalized = " ".join(instruction.lower().split())
    return any(name in normalized for name in EXECUTION_EFFECTORS) and any(
        verb in normalized.split() for verb in EXECUTION_VERBS
    )


def _format_xyz(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "] m"


class G1Pose23Encoder:
    """Encode fresh 29-DOF LowState as UnifoLM's bimanual EE state."""

    def __init__(self, urdf: Path, *, right_gripper: float, left_gripper: float) -> None:
        try:
            import pinocchio as pin
        except Exception as exc:  # pragma: no cover - GB10 dependency
            raise VLAError("pinocchio is required for UnifoLM proprioception") from exc
        if not urdf.is_file():
            raise VLAError(f"G1 URDF is missing: {urdf}")
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        missing = [name for name in BODY_JOINT_NAMES if not self.model.existJointName(name)]
        if missing:
            raise VLAError(f"G1 URDF is missing body joints: {', '.join(missing)}")
        self.left_frame = self.model.getFrameId("left_hand_palm_link")
        self.right_frame = self.model.getFrameId("right_hand_palm_link")
        self.right_gripper = float(right_gripper)
        self.left_gripper = float(left_gripper)
        if self.left_frame >= len(self.model.frames) or self.right_frame >= len(self.model.frames):
            raise VLAError("G1 URDF has no left/right hand palm frames")

    def encode(self, body_q: Sequence[float]) -> tuple[float, ...]:
        values = np.asarray(body_q, dtype=float)
        if values.shape != (29,) or not np.all(np.isfinite(values)):
            raise VLAError("robot visualization must contain 29 finite measured joints")
        q = self.pin.neutral(self.model)
        for name, value in zip(BODY_JOINT_NAMES, values, strict=True):
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1:
                raise VLAError(f"unexpected multi-DOF G1 joint: {name}")
            q[joint.idx_q] = value
        self.pin.framesForwardKinematics(self.model, self.data, q)

        def transform(frame: int) -> np.ndarray:
            pose = self.data.oMf[frame]
            result = np.eye(4)
            result[:3, :3] = pose.rotation
            result[:3, 3] = pose.translation
            return result

        # This robot has no controllable hand.  Use the selected dataset
        # profile's mean values rather than an out-of-distribution zero.  The
        # terminal never forwards predicted gripper values.
        return compose_pose23(
            transform(self.left_frame),
            transform(self.right_frame),
            right_gripper=self.right_gripper,
            left_gripper=self.left_gripper,
            waist_yaw_roll_pitch=values[12:15],
        )


@dataclass(frozen=True)
class Observation:
    image: Image.Image
    proprio: tuple[float, ...]
    frame_age_ms: float
    state_age_ms: float


class LiveObservationSource:
    def __init__(self, server: str, encoder: G1Pose23Encoder, timeout_s: float = 2.0) -> None:
        self.server = server.rstrip("/")
        self.encoder = encoder
        self.timeout_s = timeout_s
        self.last_state: dict[str, Any] | None = None

    def _get(self, path: str) -> tuple[bytes, Any]:
        request = urllib.request.Request(f"{self.server}{path}", headers={"Cache-Control": "no-cache"})
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_s)
            return response.read(), response.headers
        except (OSError, urllib.error.HTTPError) as exc:
            raise VLAError(f"GB10 observation endpoint {path} is unavailable: {exc}") from exc

    def snapshot(self) -> Observation:
        image_raw, headers = self._get("/raw-snapshot.jpg")
        image_received_at = time.monotonic()
        try:
            image = Image.open(io.BytesIO(image_raw)).convert("RGB")
            server_frame_age_ms = float(headers.get("X-G1-Frame-Age-Ms", "inf"))
        except (OSError, ValueError) as exc:
            raise VLAError("latest RGB observation is invalid") from exc

        # Read state after the image so the two observations are both bounded
        # at the end of capture.  Add state-request time to the frame age; a
        # slow request therefore fails closed instead of using an old image.
        state_raw, _ = self._get("/api/v1/arm/state")
        frame_age_ms = server_frame_age_ms + (time.monotonic() - image_received_at) * 1000.0
        try:
            state = json.loads(state_raw)
            self.last_state = state
            visual = state["visualization"]
            body_q = visual["measured_pose_rad"]
            state_age_ms = float(visual["state_age_ms"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VLAError("robot state is missing the measured 29-DOF visualization") from exc
        if not bool(visual.get("fresh")) or state_age_ms > MAX_OBSERVATION_AGE_MS:
            raise VLAError(f"robot state is stale ({state_age_ms:.1f} ms)")
        if frame_age_ms > MAX_OBSERVATION_AGE_MS:
            raise VLAError(f"RGB observation is stale ({frame_age_ms:.1f} ms)")
        return Observation(image, self.encoder.encode(body_q), frame_age_ms, state_age_ms)


class FileObservationSource:
    def __init__(self, image: Path, proprio: Sequence[float]) -> None:
        self.image = image
        self.proprio = tuple(float(value) for value in proprio)

    def snapshot(self) -> Observation:
        if len(self.proprio) != 23:
            raise VLAError("offline proprioception must contain 23 values")
        try:
            image = Image.open(self.image).convert("RGB")
        except OSError as exc:
            raise VLAError(f"could not read offline image: {self.image}") from exc
        return Observation(image, self.proprio, 0.0, 0.0)


class VisionOnlyObservationSource:
    """Use fresh RGB with profile-mean proprio for a first model smoke test."""

    def __init__(self, server: str, proprio: Sequence[float], timeout_s: float = 2.0) -> None:
        self.server = server.rstrip("/")
        self.proprio = tuple(float(value) for value in proprio)
        self.timeout_s = timeout_s

    def snapshot(self) -> Observation:
        request = urllib.request.Request(
            f"{self.server}/raw-snapshot.jpg", headers={"Cache-Control": "no-cache"}
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_s)
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
            frame_age_ms = float(response.headers.get("X-G1-Frame-Age-Ms", "inf"))
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise VLAError(f"GB10 RGB preview is unavailable: {exc}") from exc
        if frame_age_ms > MAX_OBSERVATION_AGE_MS:
            raise VLAError(f"RGB observation is stale ({frame_age_ms:.1f} ms)")
        return Observation(image, self.proprio, frame_age_ms, 0.0)


class UnifoLMRuntime:
    """Load Unitree's official model with a GB10-compatible attention backend."""

    def __init__(self, checkpoint: Path, vlm: Path, profile: str) -> None:
        self.checkpoint = checkpoint
        self.vlm_path = vlm
        self.profile = profile
        self.model: Any | None = None
        self.processor: Any | None = None
        self.stats: dict[str, Any] | None = None
        self._load_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self._load_error: Exception | None = None
        self._load_started_at: float | None = None

    def start_loading(self) -> None:
        """Warm the model immediately while leaving the task prompt responsive."""
        if self.model is not None or self._load_error is not None:
            return
        if self._load_thread is not None and self._load_thread.is_alive():
            return
        self._load_started_at = time.monotonic()
        self._load_thread = threading.Thread(
            target=self._load_in_background,
            name="unifolm-model-loader",
            daemon=True,
        )
        self._load_thread.start()

    def _load_in_background(self) -> None:
        try:
            self.load()
        except Exception:
            # load() records the original error.  The prompt and /status report
            # it on the main thread instead of printing a thread traceback.
            return

    def load_status(self) -> str:
        if self.model is not None:
            return "ready"
        if self._load_error is not None:
            return f"failed — {self._load_error}"
        if self._load_thread is not None and self._load_thread.is_alive():
            elapsed = 0.0
            if self._load_started_at is not None:
                elapsed = max(0.0, time.monotonic() - self._load_started_at)
            return f"loading in background ({elapsed:.0f}s)"
        return "not loaded"

    def load(self) -> None:
        with self._load_lock:
            if self.model is not None:
                return
            if self._load_error is not None:
                raise VLAError(f"UnifoLM model load previously failed: {self._load_error}")
            if self._load_started_at is None:
                self._load_started_at = time.monotonic()
            try:
                self._load_model()
            except Exception as exc:
                self._load_error = exc
                raise

    def _load_model(self) -> None:
        if not self.checkpoint.is_file():
            raise VLAError(f"VLA checkpoint is missing: {self.checkpoint}")
        if not (self.vlm_path / "model.safetensors.index.json").is_file():
            raise VLAError(f"VLM backbone is incomplete: {self.vlm_path}")
        # Unitree's constants module selects the 23D G1 contract from argv.
        if not any("ee_6d" in value.lower() for value in sys.argv):
            sys.argv.append("--unifolm-platform=ee_6d")
        try:
            import torch
            from omegaconf import OmegaConf
            from unifolm_vla.model.framework.__init__ import build_framework
            from unifolm_vla.model.framework.share_tools import read_mode_config
            from unifolm_vla.model.modules.vlm import QWen2_5 as qwen_module
        except Exception as exc:  # pragma: no cover - GB10 dependency
            raise VLAError(f"official UnifoLM runtime is unavailable: {exc}") from exc
        if not torch.cuda.is_available():
            raise VLAError("UnifoLM inference requires CUDA on the GB10")
        config, norm_stats = read_mode_config(self.checkpoint)
        if self.profile not in norm_stats:
            raise VLAError(
                f"normalization profile {self.profile!r} is unavailable; "
                f"choose one of {', '.join(sorted(norm_stats))}"
            )
        cfg = OmegaConf.create(config)
        cfg.framework.qwenvl.base_vlm = str(self.vlm_path)
        # Unitree's Qwen wrapper currently hard-codes FlashAttention 2 and
        # ignores its own config value.  It is not shipped in this runtime and
        # is not a safe GB10/sm_121 default.  Override only the load keyword;
        # PyTorch SDPA leaves the official architecture and weights unchanged.
        qwen_class = qwen_module.Qwen2_5_VLForConditionalGeneration
        original_from_pretrained = qwen_class.from_pretrained

        def load_vlm_with_sdpa(*args: Any, **kwargs: Any) -> Any:
            kwargs["attn_implementation"] = "sdpa"
            return original_from_pretrained(*args, **kwargs)

        cfg.framework.qwenvl.attn_implementation = "sdpa"
        with patch.object(qwen_class, "from_pretrained", new=staticmethod(load_vlm_with_sdpa)):
            model = build_framework(cfg=cfg)
        try:
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 2.5 compatibility
            state = torch.load(self.checkpoint, map_location="cpu")
        quantized = bool(cfg.framework.qwenvl.get("load_in_4bit", False))
        _load_checkpoint_state(model, state, quantized=quantized)
        model.norm_stats = norm_stats
        if quantized:
            # The VLM is already materialized on CUDA by its device map.  A
            # dtype-wide conversion would incorrectly dequantize NF4 weights;
            # moving the wrapper places the action head without changing the
            # quantized parameter representation.
            self.model = model.to("cuda").eval()
        else:
            self.model = model.to(torch.bfloat16).to("cuda").eval()
        self.processor = self.model.qwen_vl_interface.processor
        self.stats = norm_stats[self.profile]

    @staticmethod
    def _normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        low = np.asarray(stats["q01"], dtype=float)
        high = np.asarray(stats["q99"], dtype=float)
        mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
        if values.shape[-1] != low.shape[0]:
            raise VLAError(f"proprio dimension {values.shape[-1]} does not match stats {low.shape[0]}")
        return np.clip(np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values), -1, 1)

    @staticmethod
    def _unnormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        low = np.asarray(stats["q01"], dtype=float)
        high = np.asarray(stats["q99"], dtype=float)
        mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
        if values.shape[-1] != low.shape[0]:
            raise VLAError(f"action dimension {values.shape[-1]} does not match stats {low.shape[0]}")
        return np.where(mask, 0.5 * (values + 1) * (high - low + 1e-8) + low, values)

    def predict(self, observation: Observation, instruction: str) -> tuple[np.ndarray, float]:
        self.load()
        assert self.model is not None and self.processor is not None and self.stats is not None
        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover - GB10 dependency
            raise VLAError(f"UnifoLM inference dependency is unavailable: {exc}") from exc
        image = observation.image.resize((224, 224), Image.Resampling.LANCZOS)
        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f'The task is "{instruction.strip().lower()}".'},
            ],
        }
        messages = [message]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        state = self._normalize(np.asarray(observation.proprio, dtype=float), self.stats["proprio"])
        batch["state"] = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(0).to("cuda")
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            batch[key] = batch[key].to("cuda")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.predict_action(qwen_inputs=batch)
        elapsed = time.monotonic() - started
        # The official framework already detaches and converts this result to
        # NumPy.  Accept a tensor as well so the adapter remains compatible if
        # Unitree removes that conversion in a later pinned revision.
        normalized_result = output["normalized_actions"]
        if hasattr(normalized_result, "detach"):
            normalized_result = normalized_result.detach().float().cpu().numpy()
        normalized = np.asarray(normalized_result[0], dtype=float)
        action = self._unnormalize(normalized, self.stats["action"])
        parse_action_chunk(action)
        return action, elapsed


class GuardedVLAExecutor:
    """Execute only the right-arm translation through existing safety gates."""

    def __init__(
        self,
        urdf: Path,
        calibration_path: Path,
        *,
        period_s: float,
        max_waypoints: int,
    ) -> None:
        self.solver = G1RightArmIK(urdf)
        try:
            calibration = load_calibration(calibration_path)
        except (OSError, TypeError, ValueError) as exc:
            raise VLAError(f"could not load motion calibration: {exc}") from exc
        self.calibration_id = calibration.calibration_id
        points = calibration.solve_poses
        if not points:
            raise VLAError("calibration has no tabletop solve poses")
        table_z = max(float(pose.torso_point_m[2]) for pose in points)
        minimum = calibration.workspace.minimum
        maximum = calibration.workspace.maximum
        plane = Plane((0.0, 0.0, 1.0), -table_z)
        corners = (
            (minimum[0], maximum[1], table_z),
            (minimum[0], minimum[1], table_z),
            (maximum[0], minimum[1], table_z),
            (maximum[0], maximum[1], table_z),
        )
        self.calibrated_support = SupportRegion.from_ordered_corners(
            plane, corners, source="calibrated_static_tabletop"
        )
        self.period_s = period_s
        self.max_waypoints = max_waypoints
        self.transport = RosTrackingTransport(service_timeout_s=3.0)
        self.transport.start()

    def _motion_context(
        self, source: LiveObservationSource
    ) -> tuple[dict[str, Any], SupportRegion, tuple[float, ...], str]:
        state = source.last_state
        if not isinstance(state, dict):
            raise VLAError("live robot state is unavailable")
        visualization = state.get("visualization")
        if not isinstance(visualization, dict):
            raise VLAError("robot visualization is unavailable")
        support_raw = visualization.get("support_plane")
        if isinstance(support_raw, dict):
            try:
                support = SupportRegion.from_dict(support_raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise VLAError(f"invalid tabletop support plane: {exc}") from exc
        else:
            support = self.calibrated_support
        if set(support.certified_edges) != {"u_min", "u_max", "v_min", "v_max"}:
            raise VLAError("tabletop footprint is not certified on all four edges")
        measured = state.get("measured_arm_q")
        if not isinstance(measured, list) or len(measured) != 14:
            raise VLAError("robot state has no measured 14-joint arm pose")
        calibration_id = str(state.get("calibration_id") or self.calibration_id)
        return state, support, tuple(float(value) for value in measured[-7:]), calibration_id

    def _stream(
        self,
        session_id: str,
        calibration_id: str,
        path: Sequence[Sequence[float]],
        sequence: int,
    ) -> int:
        for target in path:
            self.transport.publish_target(
                session_id,
                sequence,
                calibration_id,
                target,
                pipeline_age_ms=0.0,
            )
            sequence += 1
            time.sleep(self.period_s)
        return sequence

    def _start_keepalive(
        self, session_id: str
    ) -> tuple[threading.Event, threading.Thread, list[Exception]]:
        """Keep the short robot deadman alive while inference blocks."""
        stopped = threading.Event()
        errors: list[Exception] = []

        def heartbeat() -> None:
            while not stopped.wait(0.1):
                try:
                    self.transport.heartbeat_arm(session_id)
                except Exception as exc:  # fail closed in the executor thread
                    errors.append(exc)
                    stopped.set()

        thread = threading.Thread(target=heartbeat, name="g1-vla-keepalive", daemon=True)
        thread.start()
        return stopped, thread, errors

    @staticmethod
    def _check_keepalive(errors: Sequence[Exception]) -> None:
        if errors:
            raise VLAError(f"arm keepalive failed: {errors[0]}")

    def execute(
        self,
        source: LiveObservationSource,
        runtime: UnifoLMRuntime,
        instruction: str,
    ) -> tuple[int, float]:
        _state, support, measured_q, calibration_id = self._motion_context(source)
        clearance = self.solver.plan_guided_clearance(measured_q, support_plane=support)
        if not clearance.ok or not clearance.q_path:
            raise VLAError(f"could not plan 5 cm table/hip clearance: {clearance.reason}")
        session_id = f"vla-{secrets.token_urlsafe(8)}"
        self.transport.enable_arm(session_id, calibration_id)
        keepalive_stop, keepalive_thread, keepalive_errors = self._start_keepalive(session_id)
        sequence = 0
        try:
            sequence = self._stream(
                session_id, calibration_id, clearance.q_path[1:], sequence
            )
            self._check_keepalive(keepalive_errors)
            # Re-observe and re-run the policy from the raised physical pose.
            time.sleep(max(0.25, self.period_s * 2))
            observation = source.snapshot()
            action, inference_s = runtime.predict(observation, instruction)
            self._check_keepalive(keepalive_errors)
            chunk = parse_action_chunk(action)
            _state, live_support, seed_q, live_calibration = self._motion_context(source)
            if live_calibration != calibration_id:
                raise VLAError("calibration changed after clearance")
            planned: list[tuple[float, ...]] = []
            for waypoint in chunk[: self.max_waypoints]:
                result = self.solver.solve_local_translation(
                    waypoint.right_transform(),
                    seed_q,
                    support_plane=live_support,
                    maximum_joint_step_rad=0.025,
                )
                if not result.ok or result.q_rad is None:
                    raise VLAError(f"VLA waypoint rejected by guarded IK: {result.reason}")
                seed_q = result.q_rad
                planned.append(seed_q)
            path_error = self.solver.validate_joint_path(
                (clearance.q_path[-1], *planned), support_plane=live_support
            )
            if path_error:
                raise VLAError(f"VLA joint path rejected: {path_error}")
            self._stream(session_id, calibration_id, planned, sequence)
            self._check_keepalive(keepalive_errors)
            return len(planned), inference_s
        finally:
            keepalive_stop.set()
            keepalive_thread.join(timeout=1.0)
            try:
                self.transport.stop_arm("vla_smoke_test_complete")
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models/pretrained"
    parser = argparse.ArgumentParser(description="UnifoLM-VLA task terminal")
    parser.add_argument("--checkpoint", type=Path, default=model_root / "UnifoLM-VLA-Base/checkpoints/pytorch_model.pt")
    parser.add_argument("--vlm", type=Path, default=model_root / "UnifoLM-VLM-Base")
    parser.add_argument("--profile", default="g1_stack_block")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--urdf", type=Path, default=default_urdf_path(root))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=root / "runs/localization/g1-tabletop-calibration.json",
    )
    parser.add_argument("--image", type=Path, help="offline RGB image instead of live GB10/robot state")
    parser.add_argument("--proprio-json", type=Path, help="offline JSON array with 23 proprio values")
    parser.add_argument("--once", help="run one instruction without entering the REPL")
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="preview with fresh RGB and profile-mean proprio; never requires robot arm ROS state",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--execute", action="store_true", help="run a guarded right-arm smoke test")
    parser.add_argument("--max-waypoints", type=int, default=3)
    parser.add_argument("--motion-period", type=float, default=0.15)
    return parser


def _profile_gripper_means(checkpoint: Path, profile: str) -> tuple[float, float]:
    means = _profile_proprio_means(checkpoint, profile)
    return means[18], means[19]


def _profile_proprio_means(checkpoint: Path, profile: str) -> tuple[float, ...]:
    statistics_path = checkpoint.parent.parent / "dataset_statistics.json"
    try:
        profiles = json.loads(statistics_path.read_text(encoding="utf-8"))
        means = tuple(float(value) for value in profiles[profile]["proprio"]["mean"])
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VLAError(
            f"could not read proprio means for {profile!r}: {statistics_path}"
        ) from exc
    if len(means) != 23 or not np.isfinite(means).all():
        raise VLAError(f"normalization profile {profile!r} has an invalid 23D proprio mean")
    return means


def _banner(color: bool, *, execute: bool) -> None:
    print()
    print(_color("UnifoLM", BOLD, color))
    mode = _color("GUARDED EXECUTION", GREEN, color) if execute else _color("INFERENCE ONLY", AMBER, color)
    print(f"Unitree VLA Base  ·  {mode}")
    print(_color("Type a task, or /help for commands.", DIM, color))
    print()


def _status(
    source: Any,
    runtime: UnifoLMRuntime,
    color: bool,
    executor: GuardedVLAExecutor | None,
) -> None:
    model = runtime.load_status()
    checkpoint = "ready" if runtime.checkpoint.is_file() else "missing"
    try:
        observation = source.snapshot()
        observation_text = _color(
            f"ready (RGB {observation.frame_age_ms:.0f} ms · state {observation.state_age_ms:.0f} ms)",
            GREEN,
            color,
        )
    except VLAError as exc:
        observation_text = _color(f"unavailable — {exc}", RED, color)
    print(f"model       {model}; checkpoint {checkpoint}")
    print(f"profile     {runtime.profile}")
    print(f"observation {observation_text}")
    execution = _color("guarded right arm", GREEN, color) if executor else _color("locked", AMBER, color)
    print(f"execution   {execution}")


def _run_prompt(
    source: Any,
    runtime: UnifoLMRuntime,
    prompt: str,
    color: bool,
    executor: GuardedVLAExecutor | None = None,
) -> None:
    normalized_prompt = " ".join(prompt.lower().split()).rstrip("!.,?")
    if normalized_prompt in GREETING_ONLY:
        print(
            _color("This is a robot task policy, not a general chat model.", AMBER, color)
        )
        print("Try: raise the right hand slightly while keeping it above the table")
        return
    if executor is not None and not _is_explicit_right_arm_instruction(prompt):
        raise VLAError(
            "execution requires an explicit right-hand task with a motion verb; "
            "for example: raise the right hand slightly"
        )
    if runtime.model is None:
        print(
            _color(
                f"model {runtime.load_status()}; waiting before capturing a fresh scene…",
                DIM,
                color,
            ),
            flush=True,
        )
        runtime.load()
    observation = source.snapshot()
    print(_color("reading scene and proposing motion…", DIM, color), flush=True)
    action, elapsed = runtime.predict(observation, prompt)
    chunk = parse_action_chunk(action)
    first, last = chunk[0], chunk[-1]
    current = np.asarray(observation.proprio[9:12], dtype=float)
    first_position = np.asarray(first.right_position_m, dtype=float)
    last_position = np.asarray(last.right_position_m, dtype=float)
    initial_jump = first_position - current
    chunk_delta = last_position - first_position
    print(
        f"proposal     {_color(str(len(chunk)) + ' waypoints', CYAN, color)} · "
        f"{elapsed:.2f}s inference"
    )
    print(
        "coordinates  torso frame: +x forward · +y left · +z up"
    )
    print(f"right hand   measured      {_format_xyz(current)}")
    print(
        f"             policy first  {_format_xyz(first_position)}  "
        f"Δ measured→first {_format_xyz(initial_jump)}"
    )
    print(
        f"             policy last   {_format_xyz(last_position)}  "
        f"Δ first→last {_format_xyz(chunk_delta)}"
    )
    if executor is None:
        print(
            f"execution    {_color('locked', AMBER, color)} — "
            "proposal was not sent to ROS or the robot"
        )
    else:
        if not isinstance(source, LiveObservationSource):
            raise VLAError("execution requires fresh live RGB and robot state")
        print(_color("planning clearance and guarded right-arm motion…", DIM, color), flush=True)
        moved, reinference_s = executor.execute(source, runtime, prompt)
        print(
            f"execution    {_color('complete', GREEN, color)} · "
            f"{moved} bounded VLA waypoints · {reinference_s:.2f}s re-inference"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    color = bool(sys.stdout.isatty() and not args.no_color and os.environ.get("NO_COLOR") is None)
    runtime = UnifoLMRuntime(args.checkpoint.expanduser(), args.vlm.expanduser(), args.profile)
    if args.execute and (args.vision_only or args.image):
        raise SystemExit("--execute requires live RGB and measured robot state")
    if not 1 <= args.max_waypoints <= 5:
        raise SystemExit("--max-waypoints must be between 1 and 5")
    if not 0.1 <= args.motion_period <= 0.5:
        raise SystemExit("--motion-period must be between 0.1 and 0.5 seconds")
    if bool(args.image) != bool(args.proprio_json):
        raise SystemExit("--image and --proprio-json must be supplied together")
    if args.vision_only and args.image:
        raise SystemExit("--vision-only cannot be combined with offline --image")
    if args.image:
        try:
            proprio = json.loads(args.proprio_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read --proprio-json: {exc}") from exc
        source: Any = FileObservationSource(args.image, proprio)
    elif args.vision_only:
        try:
            proprio = _profile_proprio_means(args.checkpoint.expanduser(), args.profile)
        except VLAError as exc:
            raise SystemExit(str(exc)) from exc
        source = VisionOnlyObservationSource(args.server, proprio)
    else:
        try:
            right_gripper, left_gripper = _profile_gripper_means(
                args.checkpoint.expanduser(), args.profile
            )
            encoder = G1Pose23Encoder(
                args.urdf,
                right_gripper=right_gripper,
                left_gripper=left_gripper,
            )
        except VLAError as exc:
            raise SystemExit(str(exc)) from exc
        source = LiveObservationSource(args.server, encoder)
    executor = (
        GuardedVLAExecutor(
            args.urdf,
            args.calibration,
            period_s=args.motion_period,
            max_waypoints=args.max_waypoints,
        )
        if args.execute
        else None
    )
    runtime.start_loading()
    if args.once:
        try:
            _run_prompt(source, runtime, args.once, color, executor)
        except VLAError as exc:
            print(_color(f"error: {exc}", RED, color), file=sys.stderr)
            return 1
        return 0
    _banner(color, execute=executor is not None)
    print(_color("model warming in background — /status shows progress", DIM, color))
    print()
    while True:
        try:
            value = input(_color(">>> ", BOLD, color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not value:
            continue
        if value in {"/bye", "/exit", "/quit"}:
            return 0
        if value == "/clear":
            print("\033[2J\033[H", end="")
            _banner(color, execute=executor is not None)
            continue
        if value == "/help":
            print("/status   check model, observation, and execution mode")
            print("/clear    clear the terminal")
            print("/exit     leave UnifoLM")
            if executor is None:
                print("Any other text proposes an action; nothing is sent to the robot.")
            else:
                print("Any other text requests a clearance-first guarded right-arm action.")
            continue
        if value == "/status":
            _status(source, runtime, color, executor)
            continue
        if value.startswith("/"):
            print(_color(f"unknown command: {value} (try /help)", AMBER, color))
            continue
        try:
            _run_prompt(source, runtime, value, color, executor)
        except VLAError as exc:
            print(_color(f"error: {exc}", RED, color))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
