"""
AI Waste Sorting — real-time material detection and servo-driven bin routing.

Detection and actuation are decoupled. MediaPipe calls the result callback on its
own worker thread and expects a prompt return, so the callback only enqueues a
routing request. A dedicated actuator thread drives the servos. Without the split,
the vision pipeline stalls for the full ~4s mechanical cycle.

    camera --> detector --> queue(1) --> actuator thread --> servos

Hardware: selector servo on GPIO 18 (rotates drum), release servo on GPIO 19 (tray).
Run --mock --source clip.mp4 to exercise the pipeline without a Pi.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

log = logging.getLogger("waste_sorter")

try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO = None
    GPIO_AVAILABLE = False

try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA_AVAILABLE = False



SELECTOR_PIN = 18
RELEASE_PIN = 19
PWM_FREQ_HZ = 50


ANGLE_DIVISOR = 18.0
DUTY_OFFSET = 3.0

SERVO_SETTLE_S = 0.5  
RELEASE_HOLD_S = 1.5  


BIN_ANGLES: dict[str, int] = {
    "metal": 0,
    "paper": 70,
    "plastic": 180,
}

GATE_CLOSED = 0
GATE_OPEN = 180


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_WINDOW = 10

PANEL_BG = (28, 28, 28)
TEXT = (245, 245, 245)
MUTED = (165, 165, 165)
ACCENT = (0, 220, 120)
BUSY = (60, 170, 250)




class Servos(Protocol):
    """Interface the actuator thread depends on."""

    def select_bin(self, angle: int) -> None: ...

    def release(self) -> None: ...

    def close(self) -> None: ...


class ServoController:
    """Drives the bin selector and release gate over hardware PWM."""

    def __init__(self, selector_pin: int, release_pin: int, freq_hz: int) -> None:
        if not GPIO_AVAILABLE:
            raise RuntimeError(
                "RPi.GPIO unavailable — run with --mock outside a Raspberry Pi"
            )

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(selector_pin, GPIO.OUT)
        GPIO.setup(release_pin, GPIO.OUT)

        self._selector = GPIO.PWM(selector_pin, freq_hz)
        self._release = GPIO.PWM(release_pin, freq_hz)

        self._selector.start(self._duty(0))
        self._release.start(self._duty(GATE_CLOSED))
        time.sleep(SERVO_SETTLE_S)
        self._selector.ChangeDutyCycle(0)
        self._release.ChangeDutyCycle(0)

    @staticmethod
    def _duty(angle: float) -> float:
        return angle / ANGLE_DIVISOR + DUTY_OFFSET

    def select_bin(self, angle: int) -> None:
        """Rotate the chute, wait for it to settle, then cut the pulse train.

        Holding a duty cycle indefinitely makes cheap servos hunt and buzz, which
        draws current and adds vibration to the chute. Dropping to 0% lets them
        settle mechanically.
        """
        self._selector.ChangeDutyCycle(self._duty(angle))
        time.sleep(SERVO_SETTLE_S)
        self._selector.ChangeDutyCycle(0)

    def release(self) -> None:
        self._release.ChangeDutyCycle(self._duty(GATE_OPEN))
        time.sleep(RELEASE_HOLD_S)
        self._release.ChangeDutyCycle(self._duty(GATE_CLOSED))
        time.sleep(SERVO_SETTLE_S)
        self._release.ChangeDutyCycle(0)

    def close(self) -> None:
        self._selector.stop()
        self._release.stop()
        GPIO.cleanup()


class MockServos:
    """Logs what the servos would do. Used for off-Pi runs and demo capture."""

    def select_bin(self, angle: int) -> None:
        log.info("[mock] selector -> %d deg", angle)
        time.sleep(SERVO_SETTLE_S)

    def release(self) -> None:
        log.info("[mock] gate open -> hold -> close")
        time.sleep(RELEASE_HOLD_S)

    def close(self) -> None:
        log.info("[mock] servos released")



@dataclass(frozen=True)
class SortRequest:
    material: str
    angle: int


class Sorter(threading.Thread):
    """Consumes sort requests and drives the servos, one item at a time.

    Queue depth is 1 by design. A request arriving mid-cycle is dropped rather
    than buffered — a stale routing command is worse than a missed one, because
    by the time it executes the item has already passed the chute.
    """

    def __init__(self, servos: Servos) -> None:
        super().__init__(daemon=True)
        self._servos = servos
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._current_angle: int | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._sorted: dict[str, int] = {name: 0 for name in BIN_ANGLES}
        self._last: str | None = None



    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def last_material(self) -> str | None:
        with self._lock:
            return self._last

    @property
    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._sorted)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._sorted.values())


    def submit(self, request: SortRequest) -> bool:
        """Enqueue a request. Returns False if a cycle is already in progress."""
        try:
            self._queue.put_nowait(request)
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return

            with self._lock:
                self._busy = True
            try:
                if request.angle != self._current_angle:
                    self._servos.select_bin(request.angle)
                    self._current_angle = request.angle
                self._servos.release()
                with self._lock:
                    self._sorted[request.material] += 1
                    self._last = request.material
                log.info("sorted %s", request.material)
            except Exception:
                log.exception("actuation failed for %s", request.material)
            finally:
                with self._lock:
                    self._busy = False

    def shutdown(self) -> None:
        """Send the sentinel, clearing a queued request if the slot is taken."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)




class FpsMeter:
    def __init__(self, window: int = FPS_WINDOW) -> None:
        self._window = window
        self._count = 0
        self._start = time.monotonic()
        self.value = 0.0

    def tick(self) -> None:
        self._count += 1
        if self._count % self._window == 0:
            now = time.monotonic()
            elapsed = now - self._start
            if elapsed > 0:
                self.value = self._window / elapsed
            self._start = now


class DetectionHandler:
    """Converts detection results into sort requests, with per-material cooldown."""

    def __init__(self, sorter: Sorter) -> None:
        self._sorter = sorter
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self.fps = FpsMeter()
        self.latest_result: vision.ObjectDetectorResult | None = None

    def __call__(
        self,
        result: vision.ObjectDetectorResult,
        _image: mp.Image,
        _timestamp_ms: int,
    ) -> None:
        """MediaPipe result callback. Must return promptly — no blocking work."""
        self.fps.tick()
        self.latest_result = result

        for material in self._ranked_materials(result):
            if self._on_cooldown(material):
                continue
            if self._sorter.submit(SortRequest(material, BIN_ANGLES[material])):
                self._mark_seen(material)
                log.debug("queued %s -> %d deg", material, BIN_ANGLES[material])
            break  # at most one item per frame

    @staticmethod
    def _ranked_materials(result: vision.ObjectDetectorResult) -> list[str]:
        """Known material labels in this frame, highest confidence first."""
        best: dict[str, float] = {}
        for detection in result.detections:
            for category in detection.categories:
                name = category.category_name
                if name in BIN_ANGLES:
                    score = category.score or 0.0
                    if score > best.get(name, -1.0):
                        best[name] = score
        return sorted(best, key=lambda name: best[name], reverse=True)

    def _on_cooldown(self, material: str) -> bool:
        with self._lock:
            last = self._last_seen.get(material, 0.0)
            return (time.monotonic() - last) < COOLDOWN_S

    def _mark_seen(self, material: str) -> None:
        with self._lock:
            self._last_seen[material] = time.monotonic()



class PiCamera:
    """Raspberry Pi camera via picamera2."""

    fps = 25.0

    def __init__(self, width: int, height: int) -> None:
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("picamera2 unavailable — pass --source to read a file")
        self._cam = Picamera2()
        self._cam.preview_configuration.main.size = (width, height)
        self._cam.preview_configuration.main.format = "RGB888"
        self._cam.preview_configuration.align()
        self._cam.configure("preview")
        self._cam.start()
        self._size = (width, height)

    def frames(self):
        while True:
            frame = self._cam.capture_array()
            frame = cv2.resize(frame, self._size)
            yield cv2.flip(frame, -1)

    def close(self) -> None:
        self._cam.stop()


class FileCamera:
    """Video file or USB webcam via OpenCV. Used off-Pi and for demo capture."""

    def __init__(self, source: str, width: int, height: int) -> None:
        target = int(source) if source.isdigit() else source
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open source: {source}")
        self._size = (width, height)

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield cv2.resize(frame, self._size)

    def close(self) -> None:
        self._cap.release()



def draw_boxes(frame, result: vision.ObjectDetectorResult | None) -> None:
    if result is None:
        return
    for detection in result.detections:
        box = detection.bounding_box
        x1, y1 = box.origin_x, box.origin_y
        x2, y2 = x1 + box.width, y1 + box.height
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT, 2)

        if detection.categories:
            top = detection.categories[0]
            label = f"{top.category_name} {top.score:.2f}"
            cv2.rectangle(
                frame, (x1, max(0, y1 - 22)), (x1 + 9 * len(label), y1), ACCENT, -1
            )
            cv2.putText(
                frame, label, (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
            )


def draw_panel(frame, sorter: Sorter, fps: float) -> None:
    """Stats panel — makes the demo readable without narration."""
    pw, ph = 250, 140

    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (14 + pw, 14 + ph), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

    status = "SORTING" if sorter.busy else "READY"
    status_colour = BUSY if sorter.busy else ACCENT

    cv2.putText(frame, status, (28, 46), cv2.FONT_HERSHEY_DUPLEX,
                0.68, status_colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{sorter.total} sorted", (150, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA)

    for i, (name, count) in enumerate(sorter.totals.items()):
        cv2.putText(frame, f"{name:<9}{count}", (28, 76 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.1f} FPS", (170, 144),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, MUTED, 1, cv2.LINE_AA)




def build_camera(source: str | None, width: int, height: int):
    return FileCamera(source, width, height) if source else PiCamera(width, height)


def build_servos(mock: bool) -> Servos:
    if mock or not GPIO_AVAILABLE:
        if not mock:
            log.warning("RPi.GPIO unavailable — falling back to mock servos")
        return MockServos()
    return ServoController(SELECTOR_PIN, RELEASE_PIN, PWM_FREQ_HZ)


def run(
    model: str,
    source: str | None,
    output: Path | None,
    max_results: int,
    score_threshold: float,
    width: int,
    height: int,
    mock: bool,
    preview: bool,
) -> None:
    if not Path(model).is_file():
        raise SystemExit(f"model not found: {model}")

    servos = build_servos(mock)
    sorter = Sorter(servos)
    sorter.start()

    handler = DetectionHandler(sorter)
    detector = vision.ObjectDetector.create_from_options(
        vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=score_threshold,
            result_callback=handler,
        )
    )

    camera = build_camera(source, width, height)

    writer = None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera.fps,
            (width, height),
        )
        log.info("writing annotated video to %s", output)

    log.info("running — model=%s threshold=%.2f mock=%s", model, score_threshold, mock)

    frames = 0
    try:
        for frame in camera.frames():
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                time.time_ns() // 1_000_000,
            )

            if writer or preview:
                display = frame.copy()
                draw_boxes(display, handler.latest_result)
                draw_panel(display, sorter, handler.fps.value)

                if writer:
                    writer.write(display)
                if preview:
                    cv2.imshow("waste_sorter", display)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        log.info("shutting down")
        detector.close()
        camera.close()
        if writer:
            writer.release()
        sorter.shutdown()
        sorter.join(timeout=10)
        servos.close()
        if preview:
            cv2.destroyAllWindows()
        log.info("processed %d frames — totals %s", frames, sorter.totals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time waste sorting with servo bin routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="best.tflite", help="TFLite model path")
    parser.add_argument(
        "--source", default=None,
        help="video file or webcam index; omit to use the Pi camera",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="write an annotated MP4 (for demos)",
    )
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=FRAME_HEIGHT)
    parser.add_argument(
        "--mock", action="store_true",
        help="log servo actions instead of driving GPIO",
    )
    parser.add_argument(
        "--no-preview", dest="preview", action="store_false", help="run headless",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        model=args.model,
        source=args.source,
        output=args.output,
        max_results=args.max_results,
        score_threshold=args.score_threshold,
        width=args.frame_width,
        height=args.frame_height,
        mock=args.mock,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
