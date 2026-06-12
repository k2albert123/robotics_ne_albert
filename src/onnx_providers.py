# src/onnx_providers.py
"""
ONNX Runtime provider detection and selection utilities.
"""
from __future__ import annotations
from typing import List, Optional
import onnxruntime as ort


def get_available_providers() -> List[str]:
    """
    Get list of available ONNX Runtime execution providers.
    Returns providers in order of preference (GPU first, then CPU).
    """
    available = ort.get_available_providers()
    # Preferred order: GPU providers first, then CPU
    # CUDA is fastest, DirectML is fallback for CUDA version conflicts
    preferred_order = [
        "CUDAExecutionProvider",  # Fastest for NVIDIA GPUs (requires CUDA 12.x or 13.x)
        "TensorrtExecutionProvider",  # Optimized NVIDIA inference
        "DmlExecutionProvider",  # DirectML (Windows fallback, works with AMD/NVIDIA/Intel, no CUDA version issues)
        "CPUExecutionProvider",
    ]
    
    ordered = []
    for provider in preferred_order:
        if provider in available:
            ordered.append(provider)
    
    # Add any other providers not in preferred list
    for provider in available:
        if provider not in ordered:
            ordered.append(provider)
    
    return ordered


def select_provider_interactive() -> List[str]:
    """
    Interactively prompt user to select execution provider.
    Returns list of providers to use (fallback order).
    """
    import os
    # Allow forcing provider selection via environment variable for non-interactive runs.
    forced = os.environ.get("FORCE_ONNX_PROVIDER", "").strip().lower()
    if forced == "cpu":
        return ["CPUExecutionProvider"]
    available = get_available_providers()
    
    if not available:
        print("Warning: No providers available!")
        return ["CPUExecutionProvider"]

    if available == ["CPUExecutionProvider"]:
        print("\nONNX Runtime: CPUExecutionProvider only; using CPU.")
        return ["CPUExecutionProvider"]
    
    print("\n" + "=" * 60)
    print("ONNX Runtime Execution Provider Selection")
    print("=" * 60)
    print("\nAvailable providers:")
    
    provider_info = {
        "CUDAExecutionProvider": ("NVIDIA GPU (CUDA)", "Fastest for NVIDIA GPUs (requires CUDA 12.x or 13.x)"),
        "DmlExecutionProvider": ("DirectML (Windows GPU)", "Fallback: Works with AMD/NVIDIA/Intel GPUs, no CUDA version issues"),
        "TensorrtExecutionProvider": ("NVIDIA TensorRT", "Optimized for NVIDIA GPUs"),
        "CPUExecutionProvider": ("CPU", "Compatible everywhere, slower"),
    }
    
    for i, provider in enumerate(available, 1):
        name, desc = provider_info.get(provider, (provider, ""))
        print(f"  {i}. {name} ({provider})")
        if desc:
            print(f"     {desc}")
    
    print("\nOptions:")
    print("  - Enter number to use that provider (with CPU fallback)")
    print("  - Enter 'cpu' to force CPU only")
    print("  - Enter 'auto' to use best available (default)")
    print("\nNote: If CUDA fails with DLL errors, try DirectML (option 2) - it works")
    print("      without cuDNN installation and is only slightly slower.")
    
    while True:
        choice = input("\nSelect provider [auto]: ").strip().lower()
        
        if not choice or choice == "auto":
            # Use best available (first in list)
            selected = [available[0]]
            if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
                selected.append("CPUExecutionProvider")
            return selected
        
        if choice == "cpu":
            return ["CPUExecutionProvider"]
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                selected = [available[idx]]
                # Always add CPU as fallback if not already selected
                if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
                    selected.append("CPUExecutionProvider")
                return selected
            else:
                print(f"Invalid number. Please enter 1-{len(available)}")
        except ValueError:
            print("Invalid input. Please enter a number, 'cpu', or 'auto'.")


def get_provider_display_name(providers: List[str]) -> str:
    """Get a human-readable name for the primary provider."""
    if not providers:
        return "Unknown"
    
    primary = providers[0]
    names = {
        "CUDAExecutionProvider": "NVIDIA GPU (CUDA)",
        "DmlExecutionProvider": "DirectML GPU",
        "TensorrtExecutionProvider": "NVIDIA TensorRT",
        "CPUExecutionProvider": "CPU",
    }
    return names.get(primary, primary)
