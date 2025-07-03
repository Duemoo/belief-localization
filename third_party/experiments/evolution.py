"""
Production-ready adversarial localization implementation
Fixes: memory efficiency, correct genetic operators, metric parity, device handling
"""

import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import copy
import warnings
import time, contextlib
import bisect
import wandb, time
import torch.multiprocessing as mp
import os, math
from pymoo.core.callback import Callback
from transformers import AutoModelForCausalLM, AutoTokenizer
from dsets import CounterFactDataset

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.core.repair import Repair


from util import nethook
from baselines.ft import FTHyperParams, apply_ft_to_model
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from dsets import AttributeSnippets

import random

torch.autograd.set_detect_anomaly(True)
RNG = random.Random(0xC0FFEE)

LOGIT_CLIP = 10.0          # prevents huge margins dominating
SIGMOID_STEEPNESS = 5.0    # α in σ(αΔ); change if you prefer sharper/softer curves

def worker_main(rank, gpu_id, shared_cfg, data_subset):
    torch.cuda.set_device(gpu_id)

    # deep-copy the config so each worker can tweak device/fp16 flag
    cfg = copy.deepcopy(shared_cfg)
    cfg.device = f"cuda:{gpu_id}"

    # load model *on this GPU only*
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32, device_map={"" : gpu_id}
    )
    tok   = AutoTokenizer.from_pretrained(cfg.model_name)
    tok.pad_token = tok.eos_token

    results = run_adversarial_localization_per_datapoint(
        model, tok, data_subset, cfg
    )

    # save partial result
    with open(f"{cfg.save_dir}/partial_{rank}.pkl", "wb") as f:
        pickle.dump(results, f)

def launch_parallel(dataset, cfg, gpus=None, workers_per_gpu=1):
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))

    chunks = [[] for _ in range(len(gpus) * workers_per_gpu)]
    for i, rec in enumerate(dataset):
        chunks[i % len(chunks)].append(rec)

    procs = []
    rank  = 0
    for gpu in gpus:
        for _ in range(workers_per_gpu):
            p = mp.Process(
                target=worker_main,
                args=(rank, gpu, cfg, chunks[rank])
            )
            p.start(); procs.append(p); rank += 1

    for p in procs: p.join()

    # merge partial pickles
    merged = {}
    for r in range(rank):
        with open(f"{cfg.save_dir}/partial_{r}.pkl", "rb") as f:
            merged.update(pickle.load(f))
        os.remove(f"{cfg.save_dir}/partial_{r}.pkl")
    with open(f"{cfg.save_dir}/summary_results.pkl", "wb") as f:
        pickle.dump(merged, f)
    print(f"Parallel run complete.  {len(merged)} records processed.")

def _safe_cross_entropy(logits, target):
    if logits.shape[0] != target.shape[0]:
        raise ValueError(f"Logits shape {logits.shape} does not match target shape {target.shape}")
    if logits.dim() != 2 or target.dim() != 1:
        raise ValueError(f"Expected logits to be 2D and target to be 1D, got {logits.dim()}D and {target.dim()}D")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected")
    loss = torch.nn.functional.cross_entropy(logits, target)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite CE loss")
    return loss

@contextlib.contextmanager
def stopwatch(msg: str):
    """Context-manager that prints elapsed time with a [T] prefix."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[T] {msg:<45s}: {dt:6.2f} s")

class ScalarRepair(Repair):
    """Ensure we always return exactly N distinct indices < total_params."""
    def __init__(self, total_params: int, n_select: int):
        super().__init__()
        self.total = total_params
        self.n    = n_select

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            # 1) clip to bounds
            X[i] = np.clip(X[i], 0, self.total-1).astype(int)
            # 2) deduplicate while preserving order
            uniq = []
            seen = set()
            for idx in X[i]:
                if idx not in seen:
                    uniq.append(idx); seen.add(idx)
            # 3) pad with fresh random indices if needed
            if len(uniq) < self.n:
                missing = self.n - len(uniq)
                pool = np.setdiff1d(np.arange(self.total), uniq, assume_unique=True)
                uniq.extend(np.random.choice(pool, missing, replace=False))
            # 4) truncate in case we somehow got too many
            X[i,:] = np.array(uniq[:self.n], dtype=int)
        return X

def margin_log_prob(target_logits: torch.Tensor,
                    baseline_logits: torch.Tensor) -> float:
    """
    Returns the *clipped* log-prob margin  Δ = log p_new − log p_old  ∈ [−LOGIT_CLIP, LOGIT_CLIP].
    Expects shape (vocab,) tensors (already the last-token logits of the model).
    """
    margin = (target_logits - baseline_logits).item()
    return float(np.clip(margin, -LOGIT_CLIP, LOGIT_CLIP))

def soft_success(margin: float, alpha: float = SIGMOID_STEEPNESS) -> float:
    """
    Smoothly maps a margin to (0,1) via logistic σ(αΔ).
    """
    return 1.0 / (1.0 + np.exp(-alpha * margin))


@dataclass
class AdversarialConfig:
    """Configuration for per-datapoint adversarial localization"""
    # Model configuration
    model_name: str = "EleutherAI/gpt-j-6b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    
    # Search strategy
    search_mode: str = "layer"  # "layer" or "scalar"
    target_param_count: Optional[int] = None  # Auto-computed if None
    
    # Evolution parameters
    population_size: int = 30  # Reduced for memory efficiency
    max_generations: int = 25
    crossover_prob: float = 0.4
    mutation_prob: float = 0.2
    
    # Metric weights (rewrite + generalization + locality)
    rewrite_weight: float = 1.0
    generalization_weight: float = 1.0
    locality_weight: float = 1.0
    skip_no_neighbors: bool = True  # Skip records without neighborhood prompts
    
    # Fine-tuning parameters
    ft_steps: int = 30  # Reduced for efficiency
    ft_lr: float = 1e-5  # More conservative for fp16
    ft_norm_constraint: float = 1e-4
    use_grad_scaler: bool = True  # For fp16 stability
    
    # Saving
    save_dir: str = "adversarial_results"
    save_frequency: int = 10
    
    # Evaluation parameters
    fixed_eval_size: int = 100  # Fixed size for evaluation records


class IntegerMutation:
    """Custom integer mutation for layer indices"""
    
    def __init__(self, prob: float, n_layers: int):
        self.prob = prob
        self.n_layers = n_layers
    
    def __call__(self, problem, X, **kwargs):
        X_new = X.copy()
        for i in range(len(X)):
            if np.random.random() < self.prob:
                # Random jump to different layer
                current = X[i, 0]
                choices = [j for j in range(self.n_layers) if j != current]
                if choices:
                    X_new[i, 0] = np.random.choice(choices)
        return X_new


class LayerParameterRepair(Repair):
    """Repair operator ensuring valid layer indices"""
    
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
    
    def _do(self, problem, X, **kwargs):
        """Ensure layer indices are valid"""
        for i in range(len(X)):
            X[i, 0] = np.clip(X[i, 0], 0, self.n_layers - 1)
        return X.astype(int)


class ParameterBackup:
    """Efficient parameter backup and restoration"""
    
    def __init__(self, model, param_mask: Dict[str, torch.Tensor]):
        self.backup = {}
        self.param_mask = param_mask
        
        # Store only selected parameters to save memory
        for name, param in model.named_parameters():
            if name in param_mask and param_mask[name].any():
                # Store only the selected parameters
                mask = param_mask[name]
                selected_params = param.data[mask].clone()
                self.backup[name] = (mask, selected_params)
    
    def restore(self, model):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if name in self.backup:
                mask, original_values = self.backup[name]
                param.data[mask] = original_values


class AdversarialLocalizationProblem(Problem):
    """
    Search _one_ mask that works for a fixed *set* of records.
    """

    def __init__(self,
                model,
                tokenizer,
                eval_records: List[Dict],   # fixed set chosen by caller
                config: AdversarialConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.records = eval_records
        self.config = config
        self.device = config.device
        
        # Move model to device if not already there
        if next(model.parameters()).device != torch.device(config.device):
            self.model = model.to(config.device)
        
        # Setup gradient scaler for fp16
        if config.use_fp16 and config.use_grad_scaler:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.grad_scaler = None
        
        # Check if record has required data
        for rec in self.records:
            self._validate_record(rec)
            
        # Analyze model structure
        self.layer_info = self._analyze_model_layers()
        self.total_params = sum(p.numel() for p in model.parameters())
        self.editable = []          # tuples: (name, global_start, global_end,
                                    #                   editable_start, editable_end)
        g_off = e_off = 0
        for name, p in self.model.named_parameters():
            size = p.numel()
            if (".attn." in name or ".mlp." in name) and (".bias" not in name) \
            and ("ln_" not in name and ".ln_" not in name):
                self.editable.append((name, g_off, g_off+size,
                                        e_off, e_off+size))
                e_off += size          # advance *editable* offset
            g_off += size              # always advance global offset

        self.total_editable = e_off
        cum_ends = [tpl[4] for tpl in self.editable]   # editable cumulative ends
        assert self.total_editable == cum_ends[-1], "editable space mismatch"
        
        # sanity: every random idx is mappable
        for _ in range(1000):
            i = RNG.randrange(self.total_editable)
            m  = self._create_scalar_mask(np.array([i]))
            
        SAFE_TENSORS = (".ln_", ".bias")          # never touch those
        self.editable = [tpl for tpl in self.editable
                        if not any(key in tpl[0] for key in SAFE_TENSORS)]
        self.total_editable = self.editable[-1][4]
        
        # allow only {attn,mlp}.{wq, wk, wv, wo, c_fc, c_proj}
        SAFE = (".ln_", ".bias", ".wpe", ".wte")     #  GPT-2 names
        self.editable = [tpl for tpl in self.editable
                        if not any(tok in tpl[0] for tok in SAFE)]
        self.total_editable = self.editable[-1][2] - self.editable[0][1]
        
        # Set up search space
        if config.search_mode == "layer":
            n_var = 1
            xl = np.array([0])
            xu = np.array([len(self.layer_info) - 1])
            vtype = int
        else:
            if config.target_param_count is None:
                self.target_param_count = self._estimate_rome_param_count()
            else:
                self.target_param_count = config.target_param_count
            
            n_var = self.target_param_count
            xl = np.zeros(self.target_param_count, dtype=int)
            xu = np.full(self.target_param_count, self.total_editable-1, dtype=int)
            vtype = int
        
        print(f"Search mode: {config.search_mode}")
        print(f"Total parameters: {self.total_params:,}")
        if config.search_mode == "layer":
            print(f"Available layers: {len(self.layer_info)}")
            for i, (name, count) in enumerate(self.layer_info):
                print(f"  Layer {i}: {name} ({count:,} params)")
        else:
            print(f"Target parameter count: {self.target_param_count:,}")
        
        # Single objective: maximize weighted sum of metrics
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=xl,
            xu=xu,
            vtype=vtype
        )
    
    def _validate_record(self, rec):
        """Validate that record has required fields"""
        required_fields = ["requested_rewrite"]
        for field in required_fields:
            if field not in rec:
                raise ValueError(f"Record missing required field: {field}")
        
        # Check for neighborhood prompts if locality weight > 0
        if (self.config.locality_weight > 0 and
            self.config.skip_no_neighbors and
            "neighborhood_prompts" not in rec):
            raise ValueError("Record has no neighborhood prompts but locality_weight > 0")
    
    def _analyze_model_layers(self) -> List[Tuple[str, int]]:
        """Analyze model layers and their parameter counts"""
        layer_info = []
        
        # Group parameters by transformer layer
        if hasattr(self.model.config, 'n_layer'):
            n_layers = self.model.config.n_layer
        else:
            n_layers = getattr(self.model.config, 'num_hidden_layers', 28)
        
        for layer_idx in range(n_layers):
            layer_params = 0
            layer_name = f"transformer.h.{layer_idx}"
            
            for name, param in self.model.named_parameters():
                if name.startswith(layer_name + "."):
                    layer_params += param.numel()
            
            if layer_params > 0:
                layer_info.append((layer_name, layer_params))
        
        return layer_info
    
    def _estimate_rome_param_count(self) -> int:
        """Estimate ROME-equivalent parameter count"""
        if self.layer_info:
            # Use middle layer as representative
            mid_idx = len(self.layer_info) // 2
            layer_params = self.layer_info[mid_idx][1]
            # ROME typically edits MLP out projection (~1/3 of layer)
            return layer_params // 3
        return 50000
    
    # ------------------------------------------------------------------ #
    #  Pymoo entry-point: evaluate a batch of candidate masks            #
    # ------------------------------------------------------------------ #
    def _evaluate(self, X, out, *args, **kwargs):
        vals = []
        for i, cand in enumerate(X):
            if self.config.search_mode == "layer":
                param_mask = self._create_layer_mask(int(cand[0]))
            else:
                param_mask = self._create_scalar_mask(cand.astype(int))

            r, g, l = self._score_mask(param_mask)
            if not np.isfinite([r, g, l]).all():
                raise RuntimeError(f"NaN/Inf detected  r={r} g={g} l={l}")
            vals.append(-self._combine(r, g, l))   # negative ⇒ pymoo minimises

        out["F"] = np.asarray(vals).reshape(-1, 1)

    def _create_layer_mask(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Create mask for entire layer"""
        if layer_idx >= len(self.layer_info):
            layer_idx = len(self.layer_info) - 1
        
        target_layer, _ = self.layer_info[layer_idx]
        param_mask = {}
        
        for name, param in self.model.named_parameters():
            if name.startswith(target_layer + "."):
                param_mask[name] = torch.ones_like(param, dtype=torch.bool)
            else:
                param_mask[name] = torch.zeros_like(param, dtype=torch.bool)
        
        return param_mask


    def _create_scalar_mask(self, flat_idx: np.ndarray) -> Dict[str, torch.Tensor]:
        flat_idx = np.unique(flat_idx)
        mask = {name: torch.zeros_like(self.model.get_parameter(name),
                                    dtype=torch.bool, device=self.device)
                for name, *_ in self.editable}

        cum_ends = [tpl[4] for tpl in self.editable]          # editable ends
        for idx in flat_idx:
            t = bisect.bisect_left(cum_ends, idx+1)           # +1 keeps it inclusive
            name, g_start, g_end, e_start, _ = self.editable[t]
            local = idx - e_start                             # 0 ≤ local < size
            mask[name].view(-1)[local] = True
        return mask


    
    # ------------------------------------------------------------------ #
    #  Evaluate one mask on ONE record                                   #
    # ------------------------------------------------------------------ #
    def _evaluate_on_record(self, rec, param_mask):
        backup = ParameterBackup(self.model, param_mask)
        
        # Set up gradient masking
        self._setup_gradient_masking(param_mask)
        
        metrics = self._finetune_and_evaluate_efficient(rec)
        backup.restore(self.model)
        return metrics["rewrite_score"], metrics["generalization_score"], metrics["locality_score"]
    
    def _setup_gradient_masking(self, param_mask: Dict[str, torch.Tensor]):
        """Setup gradient masking without changing requires_grad globally"""
        for name, param in self.model.named_parameters():
            if name in param_mask:
                mask = param_mask[name].to(param.device)
                param._grad_mask = mask
                param.requires_grad = mask.any().item()
            else:
                param._grad_mask = torch.zeros_like(param, dtype=torch.bool)
                param.requires_grad = False
    
    def _mask_gradients(self):
        """Apply gradient masking after backward pass"""
        for param in self.model.parameters():
            if hasattr(param, '_grad_mask') and param.grad is not None:
                param.grad *= param._grad_mask.float()
    
    def _finetune_and_evaluate_efficient(self, rec: Dict) -> Dict[str, float]:
        """Efficient fine-tuning with proper metric evaluation"""
        # Setup optimizer for selected parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            # penalise this mask so GA moves away from it
            return dict(rewrite_score=0.0, generalization_score=0.0, locality_score=0.0)
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.ft_lr, eps=1e-8)
        
        # Extract data from record
        request = rec["requested_rewrite"]
        subject = request["subject"]
        prompt = request["prompt"].format(subject)
        target_new = request["target_new"]["str"]
        
        # Fine-tuning loop
        # with stopwatch("fine-tune loop"):
        self.model.train()
        for step in range(self.config.ft_steps):
            # Tokenize prompt and target with proper device placement
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            target_ids = self.tokenizer(" " + target_new, return_tensors="pt")["input_ids"].to(self.device)
            
            if target_ids.shape[1] == 0:
                continue  # Skip if tokenization failed
            
            # Forward pass with mixed precision if enabled
            if self.config.use_fp16 and self.grad_scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :].unsqueeze(0)
                    loss = _safe_cross_entropy(logits, target_ids[0, 0].unsqueeze(0))
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self._mask_gradients()
                
                # Apply gradient clipping before unscaling
                if self.grad_scaler is not None and self.config.ft_norm_constraint > 0:
                    # --- NEW robust guard ----------------------------------------------------
                    has_fp16_grad = any(
                        (p.grad is not None) and (p.grad.dtype == torch.float16)
                        for pg in optimizer.param_groups for p in pg['params']
                    )
                    # -------------------------------------------------------------------------
                    if has_fp16_grad:                           # only then unscale + clip
                        self.grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params,
                                                    self.config.ft_norm_constraint)
                
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                # Standard precision
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :].unsqueeze(0)
                loss = _safe_cross_entropy(logits, target_ids[0, 0].unsqueeze(0))
                
                optimizer.zero_grad()
                loss.backward()
                self._mask_gradients()
                
                if self.config.ft_norm_constraint > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.ft_norm_constraint)
                
                optimizer.step()
        
        # Evaluate using proper CounterFact metrics
        # with stopwatch("CounterFact eval"):
        self.model.eval()
        with torch.no_grad():
            # metrics = self._evaluate_counterfact_metrics()
            self.record = rec          # small hack: helper expects it
            metrics = self._evaluate_counterfact_metrics()
            
        return metrics
    
    def _evaluate_counterfact_metrics(self) -> Dict[str, float]:
        """
        Returns dense scores ∈ [0,1]  for rewrite / generalisation / locality.
        Falls back gracefully if CounterFact helper throws.
        """
        try:
            cf_metrics = compute_rewrite_quality_counterfact(
                args=None,
                model=self.model,
                tok=self.tokenizer,
                record=self.record,
                snips=None,
                vec=None,
                skip_generation_tests=True
            )

            rewrite_score        = self._extract_dense_margins(cf_metrics.get("rewrite_prompts_probs",      []))
            generalization_score = self._extract_dense_margins(cf_metrics.get("paraphrase_prompts_probs",   []))
            locality_score       = self._extract_locality_dense(cf_metrics.get("neighborhood_prompts_probs",[]))

            return dict(rewrite_score=rewrite_score,
                        generalization_score=generalization_score,
                        locality_score=locality_score)

        except Exception as e:
            warnings.warn(f"CounterFact eval failed → dense fallback.  ({e})")
            return self._dense_fallback_metrics()
        
    # ------------------------------------------------------------------ #
    #  Aggregate over the fixed record-set                               #
    # ------------------------------------------------------------------ #
    def _score_mask(self, param_mask):
        # ---------- forward sanity check ------------------------------
        with torch.no_grad():
            try:
                self._setup_gradient_masking(param_mask)        # reuse
                dummy = self.tokenizer("hello", return_tensors="pt").to(self.device)
                if not torch.isfinite(self.model(**dummy).logits).all():
                    return 0.0, 0.0, 0.0        # mask is toxic -> bad fitness
            finally:
                # restore a clean state for the real evaluation
                for p in self.model.parameters():
                    p.requires_grad = False

        # ---------- true evaluation -----------------------------------
        r = g = l = 0.0
        for rec in self.records:
            ri, gi, li = self._evaluate_on_record(rec, param_mask)
            r += ri; g += gi; l += li
        n = len(self.records)
        return r/n, g/n, l/n


    # helper that was formerly _compute_fitness, now pure:
    def _combine(self, r, g, l):
        w_r = self.config.rewrite_weight
        w_g = self.config.generalization_weight
        w_l = self.config.locality_weight
        return (w_r*r + w_g*g + w_l*l) / (w_r + w_g + w_l)
        
    def _dense_fallback_metrics(self) -> Dict[str, float]:
        """
        Computes dense scores without relying on CounterFact utilities.
        Uses every prompt string already stored in the record.
        """
        req    = self.record["requested_rewrite"]
        subj   = req["subject"]
        prompt = req["prompt"].format(subj)
        new    = req["target_new"]["str"]
        old    = req["target_true"]["str"]

        # ---------- Rewrite ----------
        ln_new = self._log_prob_of_token(prompt, new)
        ln_old = self._log_prob_of_token(prompt, old)
        rewrite_score = soft_success(margin_log_prob(torch.tensor(ln_new),
                                                    torch.tensor(ln_old)))

        # ---------- Generalisation ----------
        gen_prompts = self.record.get("paraphrase_prompts", [])
        if gen_prompts:
            gens = []
            for p in gen_prompts:
                ptxt = p if isinstance(p, str) else p["prompt"]
                ln_new = self._log_prob_of_token(ptxt, new)
                ln_old = self._log_prob_of_token(ptxt, old)
                gens.append(soft_success(margin_log_prob(
                            torch.tensor(ln_new), torch.tensor(ln_old))))
            generalization_score = float(np.mean(gens))
        else:
            generalization_score = 0.0

        # ---------- Locality ----------
        neigh = self.record.get("neighborhood_prompts", [])
        if neigh:
            locs = []
            for n in neigh:
                n_prompt = n if isinstance(n, str) else n["prompt"]
                ln_new = self._log_prob_of_token(n_prompt, new)
                ln_old = self._log_prob_of_token(n_prompt, old)
                # now we *want* baseline≥new, so flip sign
                locs.append(soft_success(margin_log_prob(
                            torch.tensor(ln_old), torch.tensor(ln_new))))
            locality_score = float(np.mean(locs))
        else:
            locality_score = 1.0 if self.config.skip_no_neighbors else 0.5

        return dict(rewrite_score=rewrite_score,
                    generalization_score=generalization_score,
                    locality_score=locality_score)
    
    def _extract_success_rate(self, prompts_probs: List[Dict]) -> float:
        """Extract success rate from CounterFact probability results"""
        if not prompts_probs:
            return 0.0
        
        success_count = 0
        for prob_dict in prompts_probs:
            # Check if target_new has higher probability than target_true
            target_new_prob = prob_dict.get('target_new', float('-inf'))
            request_baseline_prob = prob_dict.get('request_baseline', float('-inf'))
            
            if target_new_prob > request_baseline_prob:
                success_count += 1
        
        return success_count / len(prompts_probs)
    
    def _log_prob_of_token(self, prompt: str, target_token: str) -> float:
        """Returns log-probability of *one* target token after the prompt."""
        # Ensure single-token target
        tid = self.tokenizer(" " + target_token,
                            return_tensors="pt").to(self.device)["input_ids"][0, 0]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]          # vocab dim
            logp  = torch.log_softmax(logits, dim=-1)[tid]
        return logp.item()

    def _extract_locality_dense(self, neigh_probs: List[Dict]) -> float:
        """
        Smooth locality score in [0,1].
        We want *baseline ≥ new*, so we flip the margin sign.
        """
        if not neigh_probs:
            return 1.0 if self.config.skip_no_neighbors else 0.5

        vals = []
        for d in neigh_probs:
            ln_new = d.get("target_new",       float('-inf'))
            ln_old = d.get("request_baseline", float('-inf'))
            # margin is ln_old - ln_new  (positive if old still preferred)
            vals.append( soft_success( margin_log_prob(ln_old, ln_new) ) )

        return float(np.mean(vals))

    
    def _extract_dense_margins(self, prompts_probs: List[Dict]) -> float:
        """
        Returns *mean soft success* over the supplied prompt-probability dicts.
        Each dict must contain 'target_new' and 'target_true' log-probs (not probs!).
        """
        if not prompts_probs:
            return 0.0

        margins = []
        for prob_dict in prompts_probs:
            # They are log-probs coming from compute_rewrite_quality_counterfact
            ln_new  = prob_dict.get('target_new',        float('-inf'))
            ln_old  = prob_dict.get('request_baseline',  float('-inf'))

            margins.append( soft_success(margin_log_prob(ln_new, ln_old)) )
        return float(np.mean(margins))
    
    
    def _evaluate_simple_metrics(self) -> Dict[str, float]:
        """Fallback simple evaluation"""
        request = self.record["requested_rewrite"]
        subject = request["subject"]
        prompt = request["prompt"].format(subject)
        target_new = request["target_new"]["str"]
        
        # Simple rewrite evaluation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        
        target_ids = self.tokenizer(" " + target_new, return_tensors="pt")["input_ids"]
        rewrite_score = 0.0
        if target_ids.shape[1] > 0:
            target_id = target_ids[0, 0]
            target_prob = torch.softmax(logits, dim=-1)[target_id].item()
            rewrite_score = target_prob
        
        return {
            'rewrite_score': rewrite_score,
            'generalization_score': rewrite_score * 0.8,  # Approximate
            'locality_score': 0.5  # Neutral
        }


class AdversarialProgressCallback(Callback):
    """Progress tracking with memory monitoring"""
    
    def __init__(self, config: AdversarialConfig, record_id: str):
        super().__init__()
        self.config = config
        self.record_id = record_id
        self.data["best_fitness"] = []
        self.data["avg_fitness"] = []
    
    def notify(self, algorithm):
        if algorithm.pop is not None:
            F = algorithm.pop.get("F").flatten()
            
            best_fitness = -F.min()
            avg_fitness = -F.mean()
            
            self.data["best_fitness"].append(best_fitness)
            self.data["avg_fitness"].append(avg_fitness)
            
            if algorithm.n_gen % 1 == 0:          # every generation
                gen_time = time.perf_counter() - algorithm.start_time
                print(f"[T] ← Generation {algorithm.n_gen:02d} finished in {gen_time:5.2f}s")
                algorithm.start_time = time.perf_counter()   # reset
            
            if algorithm.n_gen % 5 == 0:
                # Memory monitoring
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"Record {self.record_id}, Gen {algorithm.n_gen}: "
                          f"Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                          f"GPU: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
                else:
                    print(f"Record {self.record_id}, Gen {algorithm.n_gen}: "
                          f"Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")


def run_adversarial_localization_per_datapoint(
    model, tokenizer, dataset: List[Dict], config: AdversarialConfig
) -> Dict[str, Dict]:
    """
    Production-ready per-datapoint adversarial localization
    """
    results = {}
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter dataset based on config
    if config.skip_no_neighbors:
        original_size = len(dataset)
        dataset = [r for r in dataset if "neighborhood_prompts" in r and r["neighborhood_prompts"]]
        print(f"Filtered dataset: {len(dataset)}/{original_size} records have neighborhood prompts")
    
    print(f"Starting adversarial localization for {len(dataset)} datapoints")
    print(f"Model: {config.model_name} ({'FP16' if config.use_fp16 else 'FP32'})")
    print(f"Search mode: {config.search_mode}")
    print(f"Population size: {config.population_size}, Generations: {config.max_generations}")
    
    # Move model to device
    model = model.to(config.device)
    if config.use_fp16:
        model = model.half()
    
    fixed_eval_set = RNG.sample(dataset, k=min(config.fixed_eval_size, len(dataset)))
    print(f"[eval-set] Using {len(fixed_eval_set)} records for every fitness call")
    
    for i, record in enumerate(dataset):
        record_id = record.get('case_id', record.get('uuid', str(i)))
        print(f"\n{'='*60}")
        print(f"Processing record {record_id} ({i+1}/{len(dataset)})")
        print(f"Edit: {record['requested_rewrite']['subject']} -> {record['requested_rewrite']['target_new']['str']}")
        print(f"{'='*60}")
        
        try:
            # Create problem for this record
            problem = AdversarialLocalizationProblem(model, tokenizer,
                                         fixed_eval_set,  # <-- not one record
                                         config)
            
            # Setup algorithm with proper operators
            if config.search_mode == "layer":
                # Integer operators for layer selection
                sampling = IntegerRandomSampling()
                mutation = IntegerMutation(config.mutation_prob, len(problem.layer_info))
                repair = LayerParameterRepair(len(problem.layer_info))
                
                # For single variable, crossover doesn't matter much
                crossover = SBX(prob=0.0)  # Disable crossover
            else:
                print("Scalar mode detedted")
                n = problem.target_param_count
                sampling  = PermutationRandomSampling()
                crossover = OrderCrossover(prob=config.crossover_prob)
                mutation  = InversionMutation(prob=config.mutation_prob)
                repair = ScalarRepair(problem.total_editable, n)
            
            algorithm = GA(
                pop_size=config.population_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                repair=repair,
                eliminate_duplicates=False  # Allow duplicates for small search space
            )
            algorithm.start_time = time.perf_counter()
            print("[T] GA initial population being created …")
            
            # Setup progress tracking
            callback = AdversarialProgressCallback(config, str(record_id))
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', config.max_generations),
                callback=callback,
                verbose=False
            )
            
            # Extract best solution
            if hasattr(res, 'X') and res.X is not None:
                best_mask = res.X if res.X.ndim == 1 else res.X[0]
                best_fitness = -res.F if np.isscalar(res.F) else -res.F[0]
            else:
                # Fallback if optimization failed
                best_mask = np.array([0]) if config.search_mode == "layer" else np.zeros(config.target_param_count)
                best_fitness = 0.0
            
            # Create detailed results
            result_data = {
                'record_id': record_id,
                'best_mask': best_mask.tolist() if hasattr(best_mask, 'tolist') else best_mask,
                'best_fitness': float(best_fitness),
                'search_mode': config.search_mode,
                'fitness_history': callback.data["best_fitness"],
                'config': config.__dict__,
                'record_metadata': {
                    'subject': record["requested_rewrite"]["subject"],
                    'target_new': record["requested_rewrite"]["target_new"]["str"],
                    'target_true': record["requested_rewrite"]["target_true"]["str"],
                    'has_paraphrases': bool(record.get("paraphrase_prompts")),
                    'has_neighbors': bool(record.get("neighborhood_prompts")),
                    'num_paraphrases': len(record.get("paraphrase_prompts", [])),
                    'num_neighbors': len(record.get("neighborhood_prompts", []))
                }
            }
            
            # Add layer information for layer mode
            if config.search_mode == "layer":
                layer_idx = int(best_mask[0]) if hasattr(best_mask, '__len__') else int(best_mask)
                if layer_idx < len(problem.layer_info):
                    layer_name, layer_params = problem.layer_info[layer_idx]
                    result_data['selected_layer'] = {
                        'index': layer_idx,
                        'name': layer_name,
                        'param_count': layer_params
                    }
            
            # Save files
            mask_file = save_dir / f"record_{record_id}_mask.npy"
            metadata_file = save_dir / f"record_{record_id}_metadata.pkl"
            
            np.save(mask_file, best_mask)
            with open(metadata_file, 'wb') as f:
                pickle.dump(result_data, f)
            
            results[record_id] = result_data
            
            print(f"✓ Record {record_id} complete. Best fitness: {best_fitness:.4f}")
            if config.search_mode == "layer" and 'selected_layer' in result_data:
                layer_info = result_data['selected_layer']
                print(f"  Selected layer: {layer_info['index']} ({layer_info['name']}, {layer_info['param_count']:,} params)")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ Error processing record {record_id}: {e}")
            import traceback
            traceback.print_exc()
            results[record_id] = {'error': str(e)}
    
    # Save summary
    summary_file = save_dir / "summary_results.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Print final statistics
    successful = [r for r in results.values() if 'error' not in r]
    print(f"\n{'='*60}")
    print(f"Adversarial localization complete!")
    print(f"Success rate: {len(successful)}/{len(dataset)} ({100*len(successful)/len(dataset):.1f}%)")
    
    if successful and config.search_mode == "layer":
        # Analyze layer selection patterns
        layer_selections = [r.get('selected_layer', {}).get('index', -1) for r in successful]
        layer_counts = np.bincount(layer_selections)
        print(f"\nLayer selection frequency:")
        for i, count in enumerate(layer_counts):
            if count > 0:
                percentage = 100 * count / len(successful)
                print(f"  Layer {i}: {count} times ({percentage:.1f}%)")
    
    print(f"Results saved to: {save_dir}")
    return results


# Example usage
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # ← add this first
    
    # Load GPT-J 6B model
    # model_name = "EleutherAI/gpt-j-6b"
    model_name = "openai-community/gpt2"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        device_map="auto"  # For multi-GPU setups
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = CounterFactDataset("data", size=10)  # Small test set
    
    
    # Configure for production
    config = AdversarialConfig(
        model_name=model_name,
        search_mode="",                # search_mode="layer",
        population_size=20,          # Reasonable for layer search
        max_generations=100,
        
        # Balanced metric weights
        rewrite_weight=1.0,
        generalization_weight=1.0,
        locality_weight=1.0,
        skip_no_neighbors=True,      # Skip records without neighbors
        
        # Efficient fine-tuning
        ft_steps=25,
        ft_lr=1e-5,
        use_fp16=False,
        use_grad_scaler=True,
        
        # Target parameter count
        target_param_count=None,
        fixed_eval_size=100,           # Small eval set for efficiency
        
        save_dir="production_adversarial_results2"
    )
    config.use_fp16 = False
    config.use_grad_scaler = False

    
    launch_parallel(dataset, config,
                gpus=[0],        # which CUDA devices to use
                workers_per_gpu=1)     # bump to 2 if memory allows
    # Run adversarial localization
    # results = run_adversarial_localization_per_datapoint(
    #     model, tokenizer, list(dataset), config
    # )
    
    print(f"\nExperiment complete! Processed {len(results)} records.")