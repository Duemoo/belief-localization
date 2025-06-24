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

from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.core.repair import Repair

from util import nethook
from baselines.ft import FTHyperParams, apply_ft_to_model
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from dsets import AttributeSnippets

LOGIT_CLIP = 10.0          # prevents huge margins dominating
SIGMOID_STEEPNESS = 5.0    # α in σ(αΔ); change if you prefer sharper/softer curves

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
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    
    # Metric weights (rewrite + generalization + locality)
    rewrite_weight: float = 1.0
    generalization_weight: float = 1.0
    locality_weight: float = 1.0
    skip_no_neighbors: bool = True  # Skip records without neighborhood prompts
    
    # Fine-tuning parameters
    ft_steps: int = 30  # Reduced for efficiency
    ft_lr: float = 5e-5  # More conservative for fp16
    ft_norm_constraint: float = 1e-4
    use_grad_scaler: bool = True  # For fp16 stability
    
    # Saving
    save_dir: str = "adversarial_results"
    save_frequency: int = 10


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
    Memory-efficient per-datapoint adversarial localization
    """
    
    def __init__(self, model, tokenizer, record: Dict, config: AdversarialConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.record = record
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
        self._validate_record()
        
        # Analyze model structure
        self.layer_info = self._analyze_model_layers()
        self.total_params = sum(p.numel() for p in model.parameters())
        
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
            xl = np.zeros(n_var)
            xu = np.full(n_var, self.total_params - 1)
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
    
    def _validate_record(self):
        """Validate that record has required fields"""
        required_fields = ["requested_rewrite"]
        for field in required_fields:
            if field not in self.record:
                raise ValueError(f"Record missing required field: {field}")
        
        # Check for neighborhood prompts if locality weight > 0
        if (self.config.locality_weight > 0 and 
            self.config.skip_no_neighbors and
            "neighborhood_prompts" not in self.record):
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
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population efficiently"""
        fitness_values = []
        
        for i in range(X.shape[0]):
            if self.config.search_mode == "layer":
                layer_idx = int(X[i, 0])
                param_mask = self._create_layer_mask(layer_idx)
            else:
                param_indices = X[i].astype(int)
                param_mask = self._create_scalar_mask(param_indices)
            
            # Evaluate this parameter mask with memory-efficient backup
            fitness = self._evaluate_parameter_mask_efficient(param_mask)
            fitness_values.append(fitness)
        
        # Negate for minimization
        out["F"] = np.array([[-f] for f in fitness_values])
    
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
    
    def _create_scalar_mask(self, param_indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Create mask for specific parameter indices"""
        param_mask = {}
        global_idx = 0
        
        for name, param in self.model.named_parameters():
            param_size = param.numel()
            
            # Find selected indices in this parameter
            selected_locals = []
            for idx in param_indices:
                if global_idx <= idx < global_idx + param_size:
                    selected_locals.append(idx - global_idx)
            
            # Create boolean mask
            mask = torch.zeros(param_size, dtype=torch.bool, device=param.device)
            if selected_locals:
                mask[selected_locals] = True
            
            param_mask[name] = mask.view(param.shape)
            global_idx += param_size
        
        return param_mask
    
    def _evaluate_parameter_mask_efficient(self, param_mask: Dict[str, torch.Tensor]) -> float:
        """Memory-efficient evaluation with parameter backup/restore"""
        try:
            # Create backup of selected parameters
            backup = ParameterBackup(self.model, param_mask)
            
            # Set up gradient masking
            self._setup_gradient_masking(param_mask)
            
            # Fine-tune and evaluate
            metrics = self._finetune_and_evaluate_efficient()
            
            # Restore original parameters
            backup.restore(self.model)
            
            # Compute weighted fitness
            fitness = self._compute_fitness(metrics)
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating mask: {e}")
            # Ensure model is restored even on error
            try:
                backup.restore(self.model)
            except:
                pass
            return 0.0
    
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
    
    def _finetune_and_evaluate_efficient(self) -> Dict[str, float]:
        """Efficient fine-tuning with proper metric evaluation"""
        # Setup optimizer for selected parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.ft_lr, eps=1e-8)
        
        # Extract data from record
        request = self.record["requested_rewrite"]
        subject = request["subject"]
        prompt = request["prompt"].format(subject)
        target_new = request["target_new"]["str"]
        
        # Fine-tuning loop
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
                    loss = torch.nn.functional.cross_entropy(logits, target_ids[0, 0].unsqueeze(0))
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self._mask_gradients()
                
                # Apply gradient clipping before unscaling
                if self.config.ft_norm_constraint > 0:
                    self.grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.ft_norm_constraint)
                
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
            else:
                # Standard precision
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :].unsqueeze(0)
                loss = torch.nn.functional.cross_entropy(logits, target_ids[0, 0].unsqueeze(0))
                
                optimizer.zero_grad()
                loss.backward()
                self._mask_gradients()
                
                if self.config.ft_norm_constraint > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.ft_norm_constraint)
                
                optimizer.step()
        
        # Evaluate using proper CounterFact metrics
        self.model.eval()
        with torch.no_grad():
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
    
    
    def _extract_locality_score(self, neighborhood_probs: List[Dict]) -> float:
        """Extract locality preservation score"""
        if not neighborhood_probs:
            # If no neighbors, return neutral score (don't penalize or reward)
            return 0.5 if not self.config.skip_no_neighbors else 1.0
        
        preservation_count = 0
        for prob_dict in neighborhood_probs:
            # For locality, we want the request_baseline to remain higher
            target_new_prob = prob_dict.get('target_new', float('-inf'))
            request_baseline_prob = prob_dict.get('request_baseline', float('-inf'))
            
            if request_baseline_prob >= target_new_prob:
                preservation_count += 1
        
        return preservation_count / len(neighborhood_probs)
    
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
    
    def _compute_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Combine dense metrics into a single fitness value (higher = better).
        Pymoo still minimises, so caller negates later.
        """
        # locality may be missing if record lacks neighbours
        loc_w = 0.0 if ("locality_score" not in metrics and self.config.locality_weight > 0) else self.config.locality_weight
        
        w_r = self.config.rewrite_weight
        w_g = self.config.generalization_weight
        w_l = loc_w
        total_w = w_r + w_g + w_l

        if total_w == 0:
            return 0.0

        # Each metric is already in [0,1] after soft_success
        fitness = (
            w_r * metrics.get('rewrite_score',        0.0) +
            w_g * metrics.get('generalization_score',0.0) +
            w_l * metrics.get('locality_score',       0.0)
        ) / total_w
        return float(fitness)


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
    
    for i, record in enumerate(dataset):
        record_id = record.get('case_id', record.get('uuid', str(i)))
        print(f"\n{'='*60}")
        print(f"Processing record {record_id} ({i+1}/{len(dataset)})")
        print(f"Edit: {record['requested_rewrite']['subject']} -> {record['requested_rewrite']['target_new']['str']}")
        print(f"{'='*60}")
        
        try:
            # Create problem for this record
            problem = AdversarialLocalizationProblem(model, tokenizer, record, config)
            
            # Setup algorithm with proper operators
            if config.search_mode == "layer":
                # Integer operators for layer selection
                sampling = IntegerRandomSampling()
                mutation = IntegerMutation(config.mutation_prob, len(problem.layer_info))
                repair = LayerParameterRepair(len(problem.layer_info))
                
                # For single variable, crossover doesn't matter much
                crossover = SBX(prob=0.0)  # Disable crossover
            else:
                # For scalar mode (not implemented yet - would need different operators)
                raise NotImplementedError("Scalar mode needs additional implementation")
            
            algorithm = GA(
                pop_size=config.population_size,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                repair=repair,
                eliminate_duplicates=False  # Allow duplicates for small search space
            )
            
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dsets import CounterFactDataset
    
    # Load GPT-J 6B model
    model_name = "EleutherAI/gpt-j-6b"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"  # For multi-GPU setups
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = CounterFactDataset("data", size=20)  # Small test set
    
    # Configure for production
    config = AdversarialConfig(
        model_name=model_name,
        search_mode="",                # search_mode="layer",
        population_size=25,          # Reasonable for layer search
        max_generations=20,
        
        # Balanced metric weights
        rewrite_weight=1.0,
        generalization_weight=1.0,
        locality_weight=1.0,
        skip_no_neighbors=True,      # Skip records without neighbors
        
        # Efficient fine-tuning
        ft_steps=25,
        ft_lr=5e-5,
        use_fp16=True,
        use_grad_scaler=True,
        
        save_dir="production_adversarial_results"
    )
    
    # Run adversarial localization
    results = run_adversarial_localization_per_datapoint(
        model, tokenizer, list(dataset), config
    )
    
    print(f"\nExperiment complete! Processed {len(results)} records.")