# tools/fix_checkpoint_grids.py
"""
Fix grid tensor issues in existing checkpoints
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import argparse


def fix_checkpoint(checkpoint_path, output_path=None):
    """Fix grid tensor issues in checkpoint"""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' not in checkpoint:
        print("No 'model' key in checkpoint")
        return
    
    model_state = checkpoint['model']
    fixed_count = 0
    
    # Fix grid tensors
    for key in list(model_state.keys()):
        if 'grid_x' in key or 'grid_y' in key or 'grid_r' in key or 'grid_theta' in key:
            print(f"Fixing {key}")
            # Clone the tensor to ensure it's contiguous and doesn't share memory
            model_state[key] = model_state[key].clone().contiguous()
            fixed_count += 1
    
    print(f"Fixed {fixed_count} grid tensors")
    
    # Save fixed checkpoint
    if output_path is None:
        # Create backup of original
        backup_path = checkpoint_path.replace('.pth', '_backup.pth')
        print(f"Creating backup at {backup_path}")
        torch.save(checkpoint, backup_path)
        output_path = checkpoint_path
    
    torch.save(checkpoint, output_path)
    print(f"Saved fixed checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fix grid tensor issues in checkpoints')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output path for fixed checkpoint (default: overwrite original)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    fix_checkpoint(args.checkpoint, args.output)


if __name__ == '__main__':
    main()