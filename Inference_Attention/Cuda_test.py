import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version PyTorch compiled against: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")

# Test NMS (simplified)
try:
    boxes = torch.rand(10, 4).cuda() * 100
    scores = torch.rand(10).cuda()
    boxes[:, 2:] += boxes[:, :2] # Ensure x2 > x1, y2 > y1
    indices = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"NMS test on CUDA successful, indices: {indices}")
except Exception as e:
    print(f"NMS test on CUDA FAILED: {e}")