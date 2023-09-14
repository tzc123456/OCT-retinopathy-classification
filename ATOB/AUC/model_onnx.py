import torch
from model import DANNmodel
import onnx

print("ONNX 版本", onnx.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DANNmodel()
model.load_state_dict(torch.load("D:\DA_Proposed\ATOB\Pretrained method\DA\DA.pth"))
model.to(device)
x = torch.randn(1,3,224,224).to(device)
with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "resnet.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )

onnx_model = onnx.load("resnet.onnx")
onnx.checker.check_model(onnx_model)
