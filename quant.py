import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'conv_relu_gemm.onnx'
model_quant = 'conv_relu_gemm.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant)
