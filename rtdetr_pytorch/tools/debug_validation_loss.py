# tools/debug_validation_loss.py
"""
ตรวจสอบและแก้ไขปัญหา validation loss ที่สูง
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
from src.core import YAMLConfig
from src.data.coco.coco_utils_fixed import get_coco_api_from_dataset

def debug_validation_issues():
    """ตรวจสอบปัญหาใน validation pipeline"""
    
    print("🔍 ตรวจสอบปัญหา Validation Loss")
    print("=" * 60)
    
    # 1. โหลด config และ model
    cfg = YAMLConfig('configs/rtdetr/rtdetr_r50vd_face_landmark.yml')
    model = cfg.model
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor
    
    print(f"✅ โหลด model สำเร็จ")
    
    # 2. ตรวจสอบ dataset
    val_dataset = cfg.val_dataloader.dataset
    print(f"📊 Val dataset size: {len(val_dataset)}")
    
    # เช็คไม่กี่ samples
    for i in range(min(3, len(val_dataset))):
        img, target = val_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
        
        if 'landmarks' in target:
            print(f"  Landmarks range: [{target['landmarks'].min():.3f}, {target['landmarks'].max():.3f}]")
            if target['landmarks'].max() > 1.0:
                print("  ❌ ERROR: Landmarks ไม่ได้ normalize!")
                return False
        else:
            print("  ❌ ERROR: ไม่มี landmarks!")
            return False
    
    print("✅ Dataset normalization ถูกต้อง")
    
    # 3. ตรวจสอบ model output vs target alignment
    model.eval()
    val_loader = cfg.val_dataloader
    
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images[:1])  # แค่ 1 image
            
            print(f"\n🤖 Model outputs:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            
            # เช็ค landmarks alignment
            if 'pred_landmarks' in outputs:
                pred_lmks = outputs['pred_landmarks'][0, 0, :]  # First query
                target_lmks = targets[0]['landmarks'][0, :]     # First face
                
                print(f"\n📍 Landmarks comparison:")
                print(f"  Predicted: {pred_lmks[:6].tolist()}")  # แสดงแค่ 3 จุดแรก
                print(f"  Target:    {target_lmks[:6].tolist()}")
                
                diff = torch.abs(pred_lmks - target_lmks).mean()
                print(f"  Mean difference: {diff:.3f}")
                
                if diff > 0.5:
                    print("  ⚠️  ความแตกต่างสูง - อาจมีปัญหา coordinate space")
            
            break
    
    # 4. ตรวจสอบ loss computation
    print(f"\n💰 ตรวจสอบ loss computation:")
    
    with torch.no_grad():
        try:
            loss_dict = criterion(outputs, targets[:1])
            
            total_loss = 0
            for loss_name, loss_value in loss_dict.items():
                if loss_name in criterion.weight_dict:
                    weighted_loss = loss_value * criterion.weight_dict[loss_name]
                    total_loss += weighted_loss
                    print(f"  {loss_name}: {loss_value:.4f} (weight: {criterion.weight_dict[loss_name]}) = {weighted_loss:.4f}")
            
            print(f"  Total weighted loss: {total_loss:.4f}")
            
            # เช็คแต่ละ loss component
            landmark_loss = loss_dict.get('loss_landmarks', 0)
            if landmark_loss > 10:
                print(f"  ❌ Landmark loss สูงเกินไป: {landmark_loss:.4f}")
                print(f"     ลองลด weight จาก {criterion.weight_dict.get('loss_landmarks', 'N/A')} เป็น 0.5")
            
        except Exception as e:
            print(f"  ❌ Error computing loss: {e}")
            return False
    
    # 5. ตรวจสอบ COCO API conversion
    print(f"\n🎯 ตรวจสอบ COCO conversion:")
    try:
        coco_gt = get_coco_api_from_dataset(val_dataset)
        print(f"  ✅ COCO GT created: {len(coco_gt.anns)} annotations")
        
        # ทดสอบ evaluation
        orig_sizes = torch.stack([t["orig_size"] for t in targets[:1]], dim=0)
        results = postprocessor(outputs, orig_sizes)
        
        if results and len(results[0]['boxes']) > 0:
            print(f"  ✅ Postprocessor output: {len(results[0]['boxes'])} detections")
        else:
            print(f"  ⚠️  No detections from postprocessor")
            
    except Exception as e:
        print(f"  ❌ COCO conversion error: {e}")
        return False
    
    return True


def suggest_fixes():
    """แนะนำการแก้ไข"""
    
    print(f"\n🔧 แนะนำการแก้ไข:")
    print(f"=" * 60)
    
    print(f"1. 📉 ลด landmark loss weight:")
    print(f"   - เปลี่ยนจาก loss_landmarks: 5.0 เป็น 1.0")
    print(f"   - ใน config file: weight_dict")
    
    print(f"\n2. 🎯 ปรับ learning rate:")
    print(f"   - ลดจาก 0.0001 เป็น 0.00005")
    print(f"   - เพิ่ม warmup epochs")
    
    print(f"\n3. 📊 ตรวจสอบ data normalization:")
    print(f"   - ใช้ SanitizeLandmarks transform")
    print(f"   - clamp landmarks ให้อยู่ใน [0,1]")
    
    print(f"\n4. ⚖️  Progressive training:")
    print(f"   - เริ่มด้วย detection เฉพาะ (landmark weight = 0)")
    print(f"   - ค่อยๆ เพิ่ม landmark weight")
    
    print(f"\n5. 🔍 Monitor specific metrics:")
    print(f"   - ดู landmark loss แยกต่างหาก")
    print(f"   - ดู detection AP")
    print(f"   - ดู NME (Normalized Mean Error)")


if __name__ == '__main__':
    success = debug_validation_issues()
    
    if success:
        print(f"\n✅ การตรวจสอบเสร็จสิ้น")
        suggest_fixes()
    else:
        print(f"\n❌ พบปัญหาที่ต้องแก้ไขก่อน")
        suggest_fixes()