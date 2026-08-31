"""
Image classification via ONNX Runtime, serving your existing resnet50_cifar10.onnx.
Model is loaded once at import time; classify() is synchronous/CPU-bound and is
meant to be called via asyncio.to_thread() from the API layer so it doesn't
block the event loop under concurrent requests.
"""
import io
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image

from app import config

logger = logging.getLogger(__name__)

_session: ort.InferenceSession | None = None
_input_name: str | None = None


class VisionError(RuntimeError):
    pass


def _get_session() -> ort.InferenceSession:
    global _session, _input_name
    if _session is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.ONNX_INTRA_OP_THREADS
        try:
            _session = ort.InferenceSession(
                config.ONNX_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise VisionError(f"Failed to load ONNX model at {config.ONNX_MODEL_PATH}: {e}") from e
        _input_name = _session.get_inputs()[0].name
        logger.info("Loaded ONNX model from %s (input: %s)", config.ONNX_MODEL_PATH, _input_name)
    return _session


def _preprocess(image_bytes: bytes) -> np.ndarray:
    size = config.CIFAR10_INPUT_SIZE
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(config.CIFAR10_MEAN, dtype=np.float32)
    std = np.array(config.CIFAR10_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)      # add batch dim -> (1, 3, H, W)
    return arr.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def classify(image_bytes: bytes) -> dict:
    """
    Synchronous, CPU-bound. Call via asyncio.to_thread() from async endpoints.
    Returns a dict; never raises for a bad/corrupt image — returns a graceful
    degraded result instead so a single bad request can't 500 the endpoint.
    """
    try:
        session = _get_session()
        input_tensor = _preprocess(image_bytes)
    except VisionError:
        raise
    except Exception as e:
        return {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "top_k": [],
            "error": f"Could not process image: {e}",
        }

    try:
        outputs = session.run(None, {_input_name: input_tensor})
        logits = outputs[0][0]
        probs = _softmax(logits)
        top_indices = np.argsort(probs)[::-1][:3]
        top_k = [
            {"class": config.CIFAR10_CLASSES[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]
        return {
            "predicted_class": top_k[0]["class"],
            "confidence": top_k[0]["confidence"],
            "top_k": top_k,
            "error": None,
        }
    except Exception as e:
        logger.error("ONNX inference failed: %s", e)
        return {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "top_k": [],
            "error": f"Inference failed: {e}",
        }