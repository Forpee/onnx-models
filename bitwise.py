import onnx
from onnx import helper, TensorProto

# Define inputs
input1 = helper.make_tensor_value_info("input1", TensorProto.INT32, [None])
input2 = helper.make_tensor_value_info("input2", TensorProto.INT32, [None])

# Define outputs
output_and = helper.make_tensor_value_info("output_and", TensorProto.INT32, [None])
output_or = helper.make_tensor_value_info("output_or", TensorProto.INT32, [None])
output_xor = helper.make_tensor_value_info("output_xor", TensorProto.INT32, [None])

# Create nodes for bitwise ops
node_and = helper.make_node("And", ["input1", "input2"], ["output_and"])
node_or  = helper.make_node("Or",  ["input1", "input2"], ["output_or"])
node_xor = helper.make_node("Xor", ["input1", "input2"], ["output_xor"])

# Create graph
graph_def = helper.make_graph(
    nodes=[node_and, node_or, node_xor],
    name="BitwiseLogicGraph",
    inputs=[input1, input2],
    outputs=[output_and, output_or, output_xor],
)

# Create model
model_def = helper.make_model(graph_def, producer_name="bitwise-test")

# Save to file
onnx.save(model_def, "bitwise_test.onnx")