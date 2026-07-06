"""ONNX build-time utilities shared by the detector and classifier exports.

Kept separate from the serving runtime: quantization is a *build* step (run
once when producing an artifact), not something the container does per request.
"""

from pathlib import Path


def quantize_dynamic_int8(fp32_path, int8_path=None) -> Path:
    """Dynamic-INT8 quantize an FP32 ONNX model in place → a sibling .int8.onnx.

    Dynamic quantization (per Phase 4 §A.4) needs no calibration data: weights
    are stored INT8 and activations are quantized at runtime.

    CAVEAT (CNN-specific): dynamic quantization primarily targets MatMul/Gemm
    (the FC heads) and leaves Conv layers in FP32 — onnxruntime's dynamic path
    doesn't quantize Conv well. So for a ResNet the size/latency win is modest
    (the heads, not the backbone). If the CPU-latency gate isn't met, the upgrade
    is STATIC quantization (`quantize_static` + a calibration DataReader over val
    crops), which does quantize Conv. Structured so that swap is a one-function
    change here.
    """
    import tempfile

    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path) if int8_path else fp32_path.with_suffix(".int8.onnx")

    # Pre-process (symbolic shape inference + ORT graph optimization) before
    # quantizing. onnxruntime warns when this is skipped: a raw export carries
    # redundant/unoptimized nodes that make the quantizer's job harder and inflate
    # INT8 error. The cleaned graph quantizes more faithfully — measurably tighter
    # top-k agreement on the FC heads. Written to a temp file (build-only artifact).
    with tempfile.TemporaryDirectory() as td:
        pre_path = Path(td) / "preprocessed.onnx"
        quant_pre_process(input_model=str(fp32_path), output_model_path=str(pre_path))
        quantize_dynamic(
            model_input=str(pre_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
    return int8_path
