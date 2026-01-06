"""
Communication-Efficient Model Compression (NOVELTY MODULE)
==========================================================
Compresses model updates before transmission to reduce
communication overhead in federated learning.
"""

import torch
import numpy as np
from typing import Dict, Tuple, OrderedDict
from collections import OrderedDict as OD
import copy


class ModelCompressor:
    """
    Compresses model weights for efficient communication.
    
    NOVELTY: Multiple compression strategies to reduce bandwidth
    while maintaining model accuracy.
    
    Techniques:
    1. Top-K Sparsification: Send only top-K% of gradients
    2. Quantization: Reduce precision of weights
    3. Random Sparsification: Randomly sample gradients
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
        method: str = 'topk',
        quantization_bits: int = 8
    ):
        """
        Initialize the model compressor.
        
        Args:
            compression_ratio: Ratio of weights to keep (0-1)
            method: Compression method ('topk', 'random', 'quantize')
            quantization_bits: Bits for quantization (8 or 16)
        """
        self.compression_ratio = compression_ratio
        self.method = method
        self.quantization_bits = quantization_bits
        
        # Tracking
        self.compression_history = []
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0
        
    def compress(
        self, 
        weights: OrderedDict
    ) -> Tuple[OrderedDict, int]:
        """
        Compress model weights.
        
        Args:
            weights: Model weights as OrderedDict
            
        Returns:
            Tuple of (compressed_weights, compressed_size_bytes)
        """
        if self.method == 'topk':
            return self._topk_compression(weights)
        elif self.method == 'random':
            return self._random_sparsification(weights)
        elif self.method == 'quantize':
            return self._quantization(weights)
        else:
            # No compression
            size = self._calculate_size(weights)
            return weights, size
    
    def decompress(
        self, 
        compressed_weights: OrderedDict,
        original_shape: OrderedDict = None
    ) -> OrderedDict:
        """
        Decompress model weights (if applicable).
        
        For sparsification methods, zeros are filled in.
        For quantization, values are already usable.
        
        Args:
            compressed_weights: Compressed weight dictionary
            original_shape: Original shapes for reconstruction
            
        Returns:
            Decompressed weights
        """
        # For our implementation, compressed weights can be used directly
        # as we store sparse tensors in dense format with zeros
        return compressed_weights
    
    def _topk_compression(
        self, 
        weights: OrderedDict
    ) -> Tuple[OrderedDict, int]:
        """
        Top-K sparsification: Keep only the largest K% of weights.
        
        Args:
            weights: Model weights
            
        Returns:
            Tuple of (compressed_weights, size)
        """
        compressed = OD()
        k_ratio = self.compression_ratio
        
        for key, tensor in weights.items():
            flat = tensor.flatten()
            num_elements = flat.numel()
            k = max(1, int(num_elements * k_ratio))
            
            # Get top-k values and indices
            values, indices = torch.topk(flat.abs(), k)
            
            # Create sparse representation
            mask = torch.zeros_like(flat)
            mask[indices] = 1
            compressed_tensor = flat * mask
            
            compressed[key] = compressed_tensor.view(tensor.shape)
        
        # Calculate compressed size (only non-zero elements + indices)
        compressed_size = self._calculate_sparse_size(compressed)
        original_size = self._calculate_size(weights)
        
        self._log_compression(original_size, compressed_size)
        
        return compressed, compressed_size
    
    def _random_sparsification(
        self, 
        weights: OrderedDict
    ) -> Tuple[OrderedDict, int]:
        """
        Random sparsification: Randomly keep K% of weights.
        
        Args:
            weights: Model weights
            
        Returns:
            Tuple of (compressed_weights, size)
        """
        compressed = OD()
        k_ratio = self.compression_ratio
        
        for key, tensor in weights.items():
            flat = tensor.flatten()
            num_elements = flat.numel()
            
            # Random mask
            mask = torch.rand(num_elements) < k_ratio
            compressed_tensor = flat * mask.float()
            
            # Scale to maintain expected value
            compressed_tensor = compressed_tensor / k_ratio
            
            compressed[key] = compressed_tensor.view(tensor.shape)
        
        compressed_size = self._calculate_sparse_size(compressed)
        original_size = self._calculate_size(weights)
        
        self._log_compression(original_size, compressed_size)
        
        return compressed, compressed_size
    
    def _quantization(
        self, 
        weights: OrderedDict
    ) -> Tuple[OrderedDict, int]:
        """
        Quantization: Reduce precision of weights.
        
        Args:
            weights: Model weights
            
        Returns:
            Tuple of (quantized_weights, size)
        """
        quantized = OD()
        bits = self.quantization_bits
        
        for key, tensor in weights.items():
            # Get min and max for scaling
            min_val = tensor.min()
            max_val = tensor.max()
            
            # Scale to [0, 2^bits - 1]
            scale = (max_val - min_val) / (2**bits - 1)
            if scale == 0:
                scale = 1.0
            
            # Quantize
            quantized_tensor = torch.round((tensor - min_val) / scale)
            
            # Dequantize (for use in aggregation)
            dequantized = quantized_tensor * scale + min_val
            
            quantized[key] = dequantized.float()
        
        # Calculate size based on bit width
        original_size = self._calculate_size(weights)
        compressed_size = int(original_size * (bits / 32))
        
        self._log_compression(original_size, compressed_size)
        
        return quantized, compressed_size
    
    def _calculate_size(self, weights: OrderedDict) -> int:
        """Calculate size of weights in bytes."""
        total_size = 0
        for key, tensor in weights.items():
            total_size += tensor.numel() * tensor.element_size()
        return total_size
    
    def _calculate_sparse_size(self, weights: OrderedDict) -> int:
        """Calculate effective size of sparse weights."""
        total_size = 0
        for key, tensor in weights.items():
            # Count non-zero elements
            num_nonzero = torch.count_nonzero(tensor).item()
            # Each non-zero needs value (4 bytes) + index (4 bytes)
            total_size += num_nonzero * 8
        return total_size
    
    def _log_compression(self, original: int, compressed: int) -> None:
        """Log compression statistics."""
        self.total_bytes_original += original
        self.total_bytes_compressed += compressed
        
        ratio = compressed / original if original > 0 else 1.0
        
        self.compression_history.append({
            'original_bytes': original,
            'compressed_bytes': compressed,
            'ratio': ratio
        })
    
    def get_compression_statistics(self) -> Dict:
        """Get compression statistics."""
        if not self.compression_history:
            return {}
        
        return {
            'method': self.method,
            'target_ratio': self.compression_ratio,
            'total_original_bytes': self.total_bytes_original,
            'total_compressed_bytes': self.total_bytes_compressed,
            'actual_ratio': self.total_bytes_compressed / max(1, self.total_bytes_original),
            'num_compressions': len(self.compression_history),
            'bytes_saved': self.total_bytes_original - self.total_bytes_compressed
        }


class GradientAccumulator:
    """
    Accumulates residual gradients for error feedback compression.
    
    Error Feedback: Accumulated errors from previous compressions
    are added to current gradients to prevent information loss.
    """
    
    def __init__(self):
        self.residuals: Dict[str, torch.Tensor] = {}
    
    def accumulate_and_compress(
        self,
        gradients: OrderedDict,
        compressor: ModelCompressor
    ) -> OrderedDict:
        """
        Add residuals, compress, and store new residuals.
        
        Args:
            gradients: Current gradients
            compressor: Compressor to use
            
        Returns:
            Compressed gradients
        """
        # Add residuals from previous round
        adjusted_grads = OD()
        for key, grad in gradients.items():
            if key in self.residuals:
                adjusted_grads[key] = grad + self.residuals[key]
            else:
                adjusted_grads[key] = grad.clone()
        
        # Compress
        compressed, _ = compressor.compress(adjusted_grads)
        
        # Store residuals (what was lost in compression)
        for key in gradients.keys():
            self.residuals[key] = adjusted_grads[key] - compressed[key]
        
        return compressed
    
    def reset(self) -> None:
        """Reset accumulated residuals."""
        self.residuals = {}


def calculate_communication_cost(
    weights: OrderedDict,
    compressed: bool = False,
    compression_ratio: float = 0.5
) -> Dict:
    """
    Calculate communication cost for transmitting weights.
    
    Args:
        weights: Model weights
        compressed: Whether weights are compressed
        compression_ratio: Compression ratio used
        
    Returns:
        Dictionary with communication metrics
    """
    total_params = sum(t.numel() for t in weights.values())
    bytes_per_param = 4  # float32
    
    original_bytes = total_params * bytes_per_param
    
    if compressed:
        effective_bytes = int(original_bytes * compression_ratio)
    else:
        effective_bytes = original_bytes
    
    return {
        'total_parameters': total_params,
        'original_bytes': original_bytes,
        'original_mb': original_bytes / (1024 * 1024),
        'transmitted_bytes': effective_bytes,
        'transmitted_mb': effective_bytes / (1024 * 1024),
        'savings_percent': (1 - effective_bytes / original_bytes) * 100
    }


if __name__ == "__main__":
    # Test compression
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from federated.model import create_model
    
    device = torch.device("cpu")
    model = create_model(device)
    weights = model.get_weights()
    
    print("Testing Model Compression")
    print("="*50)
    
    # Test different compression methods
    for method in ['topk', 'random', 'quantize']:
        compressor = ModelCompressor(
            compression_ratio=0.5,
            method=method
        )
        
        compressed, size = compressor.compress(weights)
        stats = compressor.get_compression_statistics()
        
        print(f"\nMethod: {method}")
        print(f"Original size: {stats['total_original_bytes'] / 1024:.2f} KB")
        print(f"Compressed size: {stats['total_compressed_bytes'] / 1024:.2f} KB")
        print(f"Actual ratio: {stats['actual_ratio']:.2%}")
