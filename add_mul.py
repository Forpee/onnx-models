import onnx
from onnx import helper, TensorProto

# Define float32 input tensors
input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [None])
input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [None])

# Define outputs
output_add = helper.make_tensor_value_info("output_add", TensorProto.FLOAT, [None])
output_mul = helper.make_tensor_value_info("output_mul", TensorProto.FLOAT, [None])

# Create nodes for Add and Mul
node_add = helper.make_node("Add", ["input1", "input2"], ["output_add"])
node_mul = helper.make_node("Mul", ["input1", "input2"], ["output_mul"])

# Create the graph
graph_def = helper.make_graph(
    nodes=[node_add, node_mul],
    name="AddMulGraph",
    inputs=[input1, input2],
    outputs=[output_add, output_mul],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name="add-mul-model")

# Save the model
onnx.save(model_def, "add_mul.onnx")