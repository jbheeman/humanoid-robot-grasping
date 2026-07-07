# Unitree SDK2 Notes

`unitree_sdk2` is the likely integration path for future Unitree G1 robot access. Do not make it a hard dependency for the first tracking scaffold yet.

## Why Not Add It To Tracking Immediately

The current tracking milestone is deliberately simple:

> camera frame -> manually selected object box -> tracker update -> JSONL log

That should work with a laptop webcam, a saved video, or a future G1 camera stream. Keeping the tracker independent of the robot SDK makes it easier to test the perception loop before debugging robot connectivity.

## Where SDK2 Fits Later

Use `unitree_sdk2` or the available Unitree Python bindings when we need to:

- read the G1 onboard camera/depth stream directly
- read robot joint state
- read motor current, torque, or contact-like feedback if exposed
- command the arm or hand
- log synchronized observations, states, and actions
- run non-contact arm following
- run passive or active stopping trials

## Intended Integration Shape

Keep the tracking code modular:

```text
camera source
    -> frame
    -> detector/tracker
    -> bbox log
```

Then add a Unitree camera adapter later:

```text
unitree_sdk2 camera/depth stream
    -> CameraFrame
    -> object tracker
```

And later a control adapter:

```text
tracked object position
    -> safe target pose
    -> unitree_sdk2 arm/hand command
```

## Rule

Do not mix robot control into the tracker until the tracker works. First prove that the target plush object can be seen, boxed, tracked, and logged.

