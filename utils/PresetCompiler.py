import onnxruntime as ort
import locale
import numpy as np
from time import perf_counter

class PresetCompiler:
    """
    A class to compile and load an ONNX model using ONNX Runtime
    with the TensorRT execution provider.
    """
    def __init__(self, onnx_filename: str, cache_path: str, use_fp16: bool = True):
        """
        Initializes the PresetCompiler and loads the ONNX model into an
        InferenceSession.

        Args:
            onnx_filename (str): The path to the .onnx model file.
            use_fp16 (bool): Whether to enable FP16 precision for TensorRT.
        """
        # This is often a workaround for specific environments like Google Colab.
        locale.getpreferredencoding = lambda: "UTF-8"

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": use_fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_path,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self.ort_session = ort.InferenceSession(onnx_filename, providers=providers)

    def run_benchmark(self, iterations: int = 20):
        """
        Runs a benchmark on the loaded model to measure inference time.

        Args:
            iterations (int): The number of times to run the inference.
        """
        print("--- Starting Benchmark ---")
        # Prepare dummy input data
        input_name = self.ort_session.get_inputs()[0].name
        inp = np.random.randn(5, 3, 720, 1280).astype(np.float16)

        # Warm-up run to compile the engine
        print("Warm-up run...")
        self.ort_session.run(None, {input_name: inp})
        print("Warm-up complete.")

        print(f"Running benchmark for {iterations} iterations...")
        total_time = 0
        for i in range(iterations):
            start = perf_counter()
            self.ort_session.run(None, {input_name: inp})
            end = perf_counter()
            elapsed_ms = (end - start) * 1000
            total_time += elapsed_ms
            print(f"Iteration {i + 1}: {elapsed_ms:.2f} ms")

        print("\n--- Benchmark Finished ---")
        print(f"Average inference time: {total_time / iterations:.2f} ms")
