import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rome_main import execute_rome
from .rome_hparams import ROMEHyperParams
from util import nethook

def apply_rome_noise_to_model(
    args,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    noise_matching_strategy="random_rank1",  # New parameter
    **kwargs,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Applies noise injection that matches ROME's delta properties.
    
    Args:
        noise_matching_strategy: How to match noise to ROME delta
            - "frobenius_norm": Match Frobenius norm of the rank-1 update
            - "singular_values": Match singular values exactly
            - "component_norms": Match norms of u and v vectors separately
    """
    
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        # First, compute what ROME would do (but don't apply it)
        rome_deltas = execute_rome(args, model, tok, request, hparams)
        
        # Generate matched noise for each delta
        noise_deltas = generate_matched_noise(rome_deltas, noise_matching_strategy)

        # Apply the noise instead of ROME deltas
        with torch.no_grad():
            for w_name, (noise_u, noise_v) in noise_deltas.items():
                upd_matrix = noise_u.unsqueeze(1) @ noise_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"Matched noise successfully inserted into {list(noise_deltas.keys())}")

    return model, weights_copy


def generate_matched_noise(
    rome_deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
    strategy: str = "random_rank1"
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate noise that matches the properties of ROME deltas.
    
    Returns:
        Dictionary with same structure as rome_deltas but with matched noise
    """
    noise_deltas = {}
    
    for w_name, (rome_u, rome_v) in rome_deltas.items():
        # Create the original ROME rank-1 matrix
        rome_matrix = rome_u.unsqueeze(1) @ rome_v.unsqueeze(0)
        
    for w_name, (rome_u, rome_v) in rome_deltas.items():
        if strategy == "random_rank1":
            # Test: Does direction matter in rank-1 structure?
            noise_u = torch.randn_like(rome_u)
            noise_v = torch.randn_like(rome_v)
            noise_u = noise_u / torch.norm(noise_u) * torch.norm(rome_u)
            noise_v = noise_v / torch.norm(noise_v) * torch.norm(rome_v)
            noise_deltas[w_name] = (noise_u, noise_v)
            
        elif strategy == "full_rank":
            # Test: Does rank-1 structure matter?
            target_norm = torch.norm(rome_u) * torch.norm(rome_v)
            noise_matrix = torch.randn(rome_u.shape[0], rome_v.shape[0], 
                                     device=rome_u.device, dtype=rome_u.dtype)
            noise_matrix = noise_matrix * (target_norm / torch.norm(noise_matrix, 'fro'))
            
            # Apply as additive update (not rank-1 decomposition)
            # You'll need to modify apply_rome_noise_to_model for this case
            noise_deltas[w_name] = ("full_rank_matrix", noise_matrix)
            
        elif strategy == "scaled_random":
            # Test: Does any property matter beyond magnitude?
            # Generate completely random directions with matched total magnitude
            noise_u = torch.randn_like(rome_u)
            noise_v = torch.randn_like(rome_v)
            current_norm = torch.norm(noise_u) * torch.norm(noise_v)
            target_norm = torch.norm(rome_u) * torch.norm(rome_v)
            scale = torch.sqrt(target_norm / current_norm)
            noise_deltas[w_name] = (noise_u * scale, noise_v * scale)
            
        else:
            raise ValueError(f"Unknown noise matching strategy: {strategy}")
        
        noise_deltas[w_name] = (noise_u.detach(), noise_v.detach())
        
        # Print matching statistics for verification
        final_noise_matrix = noise_u.unsqueeze(1) @ noise_v.unsqueeze(0)
        print(f"Layer {w_name}:")
        print(f"  ROME ||ΔW||_F: {torch.norm(rome_matrix, 'fro'):.6f}")
        print(f"  Noise ||ΔW||_F: {torch.norm(final_noise_matrix, 'fro'):.6f}")
        print(f"  ROME ||u||: {torch.norm(rome_u):.6f}, ||v||: {torch.norm(rome_v):.6f}")
        print(f"  Noise ||u||: {torch.norm(noise_u):.6f}, ||v||: {torch.norm(noise_v):.6f}")
    
    return noise_deltas


# Import the upd_matrix_match_shape function from rome_main
from .rome_main import upd_matrix_match_shape