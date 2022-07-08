import onnx
onnx_model = onnx.load("../models/ddd_3dop.onnx")
graph = onnx_model.graph
nodes = graph.node
for i in range(len(nodes)):
    if(nodes[i].op_type == "Plugin"):
        nodes[i].op_type = "DCNv2"
onnx.save(onnx_model,"ddd_3dopnew.onnx")