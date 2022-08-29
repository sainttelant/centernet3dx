import onnxruntime as oxr

from onnxruntime.datasets import get_example
import torch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 测试数据
dummy_input = torch.randn(1, 3, 256, 256, device='cpu')

onnx_model = "/home/xuewei/CenterNet/CenterNet-3D_3090/models/ddd_3dop.onnx"

example_model = get_example(onnx_model)
# netron.start(example_model) 使用 netron python 包可视化网络
sess = oxr.InferenceSession(example_model)

# onnx 网络输出
onnx_out = sess.run(None, {example_model.get_inputs()[0].name: to_numpy(dummy_input)})
print(onnx_out)

#model.eval()
#ith torch.no_grad():
    # pytorch model 网络输出
    #torch_out = model(dummy_input)
	#print(torch_out)

