from object_tracking.arm_tracking.depth_tcp import DepthTcpReceiver, DepthTcpSender


def test_tcp_depth_transport_delivers_latest_frame() -> None:
    receiver = DepthTcpReceiver(0, "127.0.0.1")
    receiver.start()
    assert receiver._server is not None
    port = receiver._server.getsockname()[1]
    sender = DepthTcpSender("127.0.0.1", port)
    try:
        sender.publish(b"depth-envelope")
        item = receiver.receive(1.0)
        assert item is not None
        assert item[0] == b"depth-envelope"
    finally:
        sender.close()
        receiver.close()
