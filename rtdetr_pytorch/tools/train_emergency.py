#!/usr/bin/env python
"""
Simple RT-DETR Face Detection Training Script
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from src.core import YAMLConfig
from src.solver import TASKS

def main():
    print("🚀 Starting RT-DETR Face Detection Training")
    
    # Load config
    cfg = YAMLConfig('configs/rtdetr/emergency_final/main.yml')
    
    # Create solver  
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # Run training
    print("📝 Training for", cfg.epoches, "epochs")
    solver.fit()
    
    print("✅ Training completed!")

if __name__ == '__main__':
    main()
