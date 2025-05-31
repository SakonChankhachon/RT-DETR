# test_landmark_output.py
import torch
from core import YAMLConfig

cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
model = cfg.model
model.eval()

# Test forward
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    
print("Model outputs:", outputs.keys())
if 'pred_landmarks' in outputs:
    print("✅ Model outputs landmarks!")
    print("Landmarks shape:", outputs['pred_landmarks'].shape)
else:
    print("❌ No landmarks in output")