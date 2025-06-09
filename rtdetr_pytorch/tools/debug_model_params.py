#!/usr/bin/env python
"""
Debug script to analyze model parameters in RT-DETR
Helps identify parameter naming patterns and structure
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import re
from collections import defaultdict
from pprint import pprint

from src.core import YAMLConfig


def analyze_parameters(model, verbose=False):
    """
    Analyze model parameters and group them by prefixes
    
    Args:
        model: PyTorch model
        verbose: Whether to print detailed parameter information
    """
    # Get all parameter names
    param_names = [name for name, _ in model.named_parameters()]
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Group parameters by prefix
    prefix_groups = defaultdict(list)
    for name in param_names:
        # Extract main component (first part of parameter name)
        prefix = name.split('.')[0]
        prefix_groups[prefix].append(name)
    
    # Print parameter groups
    print("\nParameter groups by main component:")
    for prefix, names in prefix_groups.items():
        params_count = sum(p.numel() for n, p in model.named_parameters() 
                          if n in names and p.requires_grad)
        print(f"\n{prefix}: {len(names)} parameters, {params_count:,} elements")
        
        if verbose:
            for name in sorted(names):
                param = dict(model.named_parameters())[name]
                print(f"  - {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    # Look specifically for landmark-related parameters
    landmark_params = [name for name in param_names if 'landmark' in name]
    
    print("\n" + "="*80)
    print("LANDMARK-RELATED PARAMETERS:")
    print("="*80)
    
    if landmark_params:
        # Group landmark parameters by pattern
        landmark_groups = defaultdict(list)
        for name in landmark_params:
            # Extract pattern (e.g., dec_landmark_heads, enc_landmark_head)
            pattern = re.match(r'([^.]+\.[^.]+)', name)
            if pattern:
                group = pattern.group(1)
            else:
                group = name.split('.')[0]
            landmark_groups[group].append(name)
        
        # Print landmark parameter groups
        for group, names in landmark_groups.items():
            params_count = sum(p.numel() for n, p in model.named_parameters() 
                              if n in names and p.requires_grad)
            print(f"\n{group}: {len(names)} parameters, {params_count:,} elements")
            
            if verbose:
                for name in sorted(names):
                    param = dict(model.named_parameters())[name]
                    print(f"  - {name}: {param.shape}, requires_grad={param.requires_grad}")
    else:
        print("\nNo parameters containing 'landmark' found!")
        
        # Try to find any parameters that might be related to landmarks
        potential_landmark_params = [
            name for name in param_names 
            if any(term in name.lower() for term in ['point', 'keypoint', 'face', 'head'])
        ]
        
        if potential_landmark_params:
            print("\nPotential landmark-related parameters:")
            for name in sorted(potential_landmark_params):
                param = dict(model.named_parameters())[name]
                print(f"  - {name}: {param.shape}")
    
    # Test various regex patterns
    print("\n" + "="*80)
    print("TESTING REGEX PATTERNS:")
    print("="*80)
    
    patterns = [
        '.*landmark.*',
        'dec_landmark.*',
        '.*landmark_head.*',
        'decoder.*',
        'decoder.dec_landmark.*'
    ]
    
    for pattern in patterns:
        matching_params = [name for name in param_names if re.match(pattern, name)]
        print(f"\nPattern '{pattern}': {len(matching_params)} matches")
        if matching_params and verbose:
            for name in sorted(matching_params)[:10]:  # Show only first 10
                print(f"  - {name}")
            if len(matching_params) > 10:
                print(f"  ... and {len(matching_params) - 10} more")


def main(args):
    """Main function"""
    # Load configuration
    cfg = YAMLConfig(args.config)
    
    # Print configuration details
    print(f"Loading model from config: {args.config}")
    
    # Create model
    model = cfg.model
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model type: {type(model).__name__}")
    
    # Analyze parameters
    analyze_parameters(model, args.verbose)
    
    # Try to load optimizer to see what happens
    try:
        print("\nTrying to create optimizer...")
        optimizer = cfg.optimizer
        print("✅ Optimizer created successfully!")
        
        # Print optimizer parameter groups
        print("\nOptimizer parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            params_count = sum(p.numel() for p in group['params'])
            print(f"Group {i}: {params_count:,} elements, lr={group['lr']}")
    except Exception as e:
        print(f"\n❌ Failed to create optimizer: {str(e)}")
        
        # Get the detailed error
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug RT-DETR model parameters')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed parameter information')
    
    args = parser.parse_args()
    main(args)
