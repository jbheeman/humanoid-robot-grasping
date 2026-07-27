"""Small, lazy ROS 2 runtime and Unitree request/response primitives.

Nothing in this module imports :mod:`rclpy` or generated ROS messages at
module-import time.  GB10-only tools and unit tests can therefore import the
project without a ROS installation.  Robot processes either let
``Ros2NodeRunner`` own an isolated ROS context or inject an existing node.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional, Sequence


class Ros2Unavailable(RuntimeError):
    """ROS 2 or the generated Unitree messages are unavailable."""


class Ros2RpcTimeout(TimeoutError):
    """A Unitree topic-paired API request did not receive its response."""


@dataclass(frozen=True)
class Ros2Bindings:
    """Late-bound ROS modules and generated Unitree message classes."""

    rclpy: Any
    context_type: type
    executor_type: type
    low_cmd_type: type
    low_state_type: type
    request_type: type
    response_type: type

    @classmethod
    def load(cls) -> "Ros2Bindings":
        """Import the ROS bindings only inside a robot-side startup path."""

        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import MultiThreadedExecutor
            from unitree_api.msg import Request, Response
            from unitree_hg.msg import LowCmd, LowState
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - robot image only
            raise Ros2Unavailable(
                "ROS 2 and the generated unitree_hg/unitree_api messages are required; "
                "source the robot ROS workspace before starting this process"
            ) from exc
        return cls(
            rclpy=rclpy,
            context_type=Context,
            executor_type=MultiThreadedExecutor,
            low_cmd_type=LowCmd,
            low_state_type=LowState,
            request_type=Request,
            response_type=Response,
        )


class Ros2NodeRunner:
    """Own one ROS node and executor, or adapt an injected shared node.

    Owned nodes use a private ``rclpy`` context.  Closing one service therefore
    cannot shut down another ROS node in the same Python process.
    """

    def __init__(
        self,
        node_name: str,
        *,
        bindings: Optional[Ros2Bindings] = None,
        node: Optional[object] = None,
        ros_args: Optional[Sequence[str]] = None,
    ) -> None:
        if not node_name or not node_name.strip():
            raise ValueError("node_name is required")
        self.node_name = node_name
        self._bindings = bindings
        self._node = node
        self._ros_args = None if ros_args is None else list(ros_args)
        self._owns_node = node is None
        self._context: Optional[object] = None
        self._executor: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._prepared = False
        self._started = False
        self._lock = threading.Lock()

    @property
    def bindings(self) -> Ros2Bindings:
        if self._bindings is None:
            raise RuntimeError("ROS node runner has not been started")
        return self._bindings

    @property
    def node(self) -> object:
        if self._node is None or not self._started:
            raise RuntimeError("ROS node runner has not been started")
        return self._node

    @property
    def owns_node(self) -> bool:
        return self._owns_node

    def start(self) -> object:
        with self._lock:
            if self._started:
                return self.node
            node = self._prepare_locked()
            if self._owns_node:
                assert self._executor is not None
                self._thread = threading.Thread(
                    target=self._executor.spin,
                    daemon=True,
                    name=f"{self.node_name}-ros-executor",
                )
                self._thread.start()
            self._started = True
            return node

    def prepare(self) -> object:
        """Create the node without spinning so callers can add entities first."""

        with self._lock:
            return self._prepare_locked()

    def _prepare_locked(self) -> object:
        if self._prepared:
            assert self._node is not None
            return self._node
        bindings = self._bindings or Ros2Bindings.load()
        self._bindings = bindings
        if self._node is None:
            context = bindings.context_type()
            bindings.rclpy.init(args=self._ros_args, context=context)
            try:
                # Foxy Fast DDS can discover Jazzy's newer automatic
                # parameter/type-description services, but intermittently
                # fails to deserialize their metadata and eventually
                # aborts with std::bad_alloc.  Project nodes do not use
                # ROS parameters or rosout, so omit those cross-distro
                # readers entirely.
                node = bindings.rclpy.create_node(
                    self.node_name,
                    context=context,
                    enable_rosout=False,
                    start_parameter_services=False,
                )
                executor = bindings.executor_type(context=context, num_threads=2)
                executor.add_node(node)
            except Exception:
                bindings.rclpy.shutdown(context=context)
                raise
            self._context = context
            self._node = node
            self._executor = executor
        self._prepared = True
        return self._node

    def close(self) -> None:
        with self._lock:
            if not self._prepared:
                return
            self._started = False
            self._prepared = False
            if not self._owns_node:
                return
            executor = self._executor
            node = self._node
            context = self._context
            thread = self._thread
            self._executor = None
            self._node = None
            self._context = None
            self._thread = None

        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if thread is not None:
            thread.join(timeout=1.5)
        if node is not None:
            node.destroy_node()
        if context is not None and self.bindings.rclpy.ok(context=context):
            self.bindings.rclpy.shutdown(context=context)


@dataclass(frozen=True)
class Ros2RpcResult:
    """One identity-correlated Unitree API response."""

    request_id: int
    api_id: int
    status_code: int
    data: str
    binary: bytes


@dataclass
class _PendingResponse:
    event: threading.Event
    message: Optional[object] = None


class Ros2RequestResponseClient:
    """Correlate Unitree Request/Response topics by ``identity.id``."""

    def __init__(
        self,
        node: object,
        *,
        request_type: type,
        response_type: type,
        request_topic: str,
        response_topic: str,
        qos_depth: int = 1,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if qos_depth <= 0:
            raise ValueError("qos_depth must be positive")
        if not request_topic or not response_topic:
            raise ValueError("request and response topics are required")
        self._node = node
        self._request_type = request_type
        self._monotonic_ns = monotonic_ns
        self._pending: dict[int, _PendingResponse] = {}
        self._lock = threading.Lock()
        self._last_request_id = 0
        self._closed = False
        self._publisher = node.create_publisher(request_type, request_topic, qos_depth)
        self._subscription = node.create_subscription(
            response_type,
            response_topic,
            self._response_callback,
            qos_depth,
        )

    def call(
        self,
        *,
        api_id: int,
        parameter: str = "",
        binary: bytes = b"",
        timeout_s: float = 1.0,
    ) -> Ros2RpcResult:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        request = self._request_type()
        with self._lock:
            if self._closed:
                raise RuntimeError("ROS request/response client is closed")
            request_id = max(int(self._monotonic_ns()), self._last_request_id + 1)
            self._last_request_id = request_id
            pending = _PendingResponse(event=threading.Event())
            self._pending[request_id] = pending
        request.header.identity.id = request_id
        request.header.identity.api_id = int(api_id)
        request.parameter = str(parameter)
        request.binary = list(binary)
        try:
            self._publisher.publish(request)
        except Exception:
            with self._lock:
                self._pending.pop(request_id, None)
            raise

        if not pending.event.wait(timeout_s):
            with self._lock:
                self._pending.pop(request_id, None)
            raise Ros2RpcTimeout(
                f"ROS API {api_id} timed out after {timeout_s:.3f}s "
                f"(request identity {request_id})"
            )

        with self._lock:
            self._pending.pop(request_id, None)
            response = pending.message
        if response is None:  # close() wakes every waiter without a message
            raise RuntimeError("ROS request/response client closed while a call was pending")
        status = int(response.header.status.code)
        response_binary = bytes(int(value) & 0xFF for value in getattr(response, "binary", ()))
        return Ros2RpcResult(
            request_id=request_id,
            api_id=int(api_id),
            status_code=status,
            data=str(getattr(response, "data", "")),
            binary=response_binary,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            waiters = tuple(self._pending.values())
            self._pending.clear()
        for pending in waiters:
            pending.event.set()
        self._destroy("destroy_subscription", self._subscription)
        self._destroy("destroy_publisher", self._publisher)

    def _response_callback(self, message: object) -> None:
        try:
            request_id = int(message.header.identity.id)
        except (AttributeError, TypeError, ValueError):
            return
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.message is not None:
                return
            pending.message = message
            pending.event.set()

    def _destroy(self, method_name: str, entity: object) -> None:
        destroy = getattr(self._node, method_name, None)
        if callable(destroy):
            destroy(entity)
