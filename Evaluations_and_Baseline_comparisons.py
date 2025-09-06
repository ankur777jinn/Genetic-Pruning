"""
Evaluation module for D-PRUNER with baseline comparisons
Implements SparseGPT, LLM-Pruner, and Magnitude Pruning baselines
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import math

logger = logging.getLogger(__name__)


class MagnitudePruner:
    """Magnitude-based pruning baseline"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def prune(self, sparsity_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """Prune weights based on magnitude"""
        pruning_masks = {}
        
        for name, param in self.model.named_parameters():
            if self._should_prune_layer(name):
                # Compute magnitude-based importance
                importance = torch.abs(param.data)
                
                # Create mask based on magnitude
                flat_importance = importance.flatten()
                num_params = flat_importance.numel()
                num_to_prune = int(num_params * sparsity_ratio)
                
                if num_to_prune > 0:
                    threshold = torch.topk(flat_importance, num_to_prune, largest=False).values.max()
                    mask = (importance > threshold).float()
                else:
                    mask = torch.ones_like(importance)
                
                pruning_masks[name] = mask
                param.data *= mask
        
        return pruning_masks
    
    def _should_prune_layer(self, name: str) -> bool:
        """Determine if a layer should be pruned"""
        target_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return any(layer in name for layer in target_layers)


class SparseGPTSimulator:
    """Simplified SparseGPT implementation"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        
    def prune(
        self, 
        calibration_data: DataLoader, 
        sparsity_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """SparseGPT-style pruning with iterative weight updates"""
        logger.info(f"Running SparseGPT pruning with {sparsity_ratio*100}% sparsity")
        
        pruning_masks = {}
        
        # Process each layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and self._should_prune_layer(name):
                mask = self._prune_linear_layer(module, calibration_data, sparsity_ratio)
                pruning_masks[f"{name}.weight"] = mask
        
        return pruning_masks
    
    def _prune_linear_layer(
        self, 
        layer: nn.Linear, 
        calibration_data: DataLoader,
        sparsity_ratio: float
    ) -> torch.Tensor:
        """Prune a single linear layer using SparseGPT approach"""
        weight = layer.weight.data
        
        # Collect activations for Hessian approximation
        activations = self._collect_activations(layer, calibration_data)
        
        # Compute approximate Hessian
        H = self._compute_hessian_approx(activations)
        
        # Iterative pruning with weight updates
        mask = torch.ones_like(weight)
        num_to_prune = int(weight.numel() * sparsity_ratio)
        
        for _ in range(num_to_prune):
            # Find least important weight
            importance = self._compute_importance(weight, H, mask)
            min_idx = torch.argmin(importance[mask.bool()])
            
            # Convert to 2D index
            flat_indices = torch.nonzero(mask.flatten()).flatten()
            actual_idx = flat_indices[min_idx]
            row, col = divmod(actual_idx.item(), weight.size(1))
            
            # Prune weight and update others
            mask[row, col] = 0
            if H[col, col] > 0:
                delta = weight[row, col] / H[col, col]
                weight[row, :] -= delta * H[col, :]
            
            weight[row, col] = 0
        
        return mask
    
    def _collect_activations(self, layer: nn.Linear, calibration_data: DataLoader) -> torch.Tensor:
        """Collect activations for Hessian computation"""
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(input[0].detach())
        
        handle = layer.register_forward_hook(hook_fn)
        
        self.model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
        
        handle.remove()
        
        if activations:
            return torch.cat(activations, dim=0)
        return torch.empty(0)
    
    def _compute_hessian_approx(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute approximate Hessian matrix"""
        if activations.numel() == 0:
            return torch.eye(1)
        
        # Flatten activations to 2D: [batch_size * seq_len, hidden_size]
        if activations.dim() > 2:
            activations = activations.view(-1, activations.size(-1))
        
        # Compute covariance matrix as Hessian approximation
        mean = activations.mean(dim=0, keepdim=True)
        centered = activations - mean
        H = torch.mm(centered.T, centered) / (activations.size(0) - 1)
        
        # Add small diagonal for numerical stability
        H += 1e-6 * torch.eye(H.size(0), device=H.device)
        
        return H
    
    def _compute_importance(self, weight: torch.Tensor, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute weight importance scores"""
        # Compute importance as w^2 / (2 * H_ii)
        diag_H = torch.diag(H).unsqueeze(0).expand_as(weight)
        importance = (weight ** 2) / (2 * diag_H + 1e-8)
        
        # Mask out already pruned weights
        importance = importance * mask
        importance[mask == 0] = float('inf')  # Already pruned
        
        return importance
    
    def _should_prune_layer(self, name: str) -> bool:
        """Determine if a layer should be pruned"""
        target_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return any(layer in name for layer in target_layers)


class LLMPrunerSimulator:
    """Simplified LLM-Pruner implementation using gradients"""
    
    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def prune(
        self, 
        calibration_data: DataLoader,
        sparsity_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Gradient-based pruning following LLM-Pruner approach"""
        logger.info(f"Running LLM-Pruner with {sparsity_ratio*100}% sparsity")
        
        # Compute gradient-based importance
        importance_scores = self._compute_gradient_importance(calibration_data)
        
        # Create pruning masks
        pruning_masks = {}
        for name, param in self.model.named_parameters():
            if name in importance_scores:
                importance = importance_scores[name]
                mask = self._create_structured_mask(importance, sparsity_ratio)
                pruning_masks[name] = mask
                param.data *= mask
        
        return pruning_masks
    
    def _compute_gradient_importance(self, calibration_data: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute importance based on gradients during fine-tuning"""
        self.model.train()
        importance_scores = {}
        
        # Initialize gradient accumulators
        for name, param in self.model.named_parameters():
            if self._should_prune_layer(name):
                importance_scores[name] = torch.zeros_like(param)
        
        num_samples = 0
        
        for batch in tqdm(calibration_data, desc="Computing gradient importance"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward and backward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in self.model.named_parameters():
                if name in importance_scores and param.grad is not None:
                    importance_scores[name] += torch.abs(param.grad)
            
            num_samples += batch["input_ids"].size(0)
            self.model.zero_grad()
        
        # Average importance scores
        for name in importance_scores:
            importance_scores[name] /= num_samples
        
        return importance_scores
    
    def _create_structured_mask(self, importance: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Create structured pruning mask"""
        if importance.dim() == 1:
            # For 1D tensors (bias), use unstructured pruning
            return self._create_unstructured_mask(importance, sparsity_ratio)
        
        # For 2D tensors (weights), consider structured pruning by removing entire rows/columns
        # For simplicity, we'll use unstructured here, but LLM-Pruner typically does structured
        return self._create_unstructured_mask(importance, sparsity_ratio)
    
    def _create_unstructured_mask(self, importance: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Create unstructured pruning mask"""
        flat_importance = importance.flatten()
        num_params = flat_importance.numel()
        num_to_prune = int(num_params * sparsity_ratio)
        
        if num_to_prune > 0:
            threshold = torch.topk(flat_importance, num_to_prune, largest=False).values.max()
            mask = (importance > threshold).float()
        else:
            mask = torch.ones_like(importance)
        
        return mask
    
    def _should_prune_layer(self, name: str) -> bool:
        """Determine if a layer should be pruned"""
        target_layers = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return any(layer in name for layer in target_layers)


class ModelEvaluator:
    """Comprehensive model evaluation suite"""
    
    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_perplexity(self, test_texts: List[str], batch_size: int = 8) -> float:
        """Evaluate model perplexity"""
        from d_pruner import DomainSpecificDataset  # Import from main module
        
        dataset = DomainSpecificDataset(test_texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating perplexity"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        return perplexity
    
    def evaluate_generation_quality(
        self, 
        prompts: List[str], 
        max_length: int = 100,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Evaluate text generation quality"""
        self.model.eval()
        generated_texts = []
        
        for prompt in tqdm(prompts, desc="Generating text"):
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text)
        
        return generated_texts
    
    def compute_sparsity_stats(self, pruning_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute detailed sparsity statistics"""
        stats = {}
        total_params = 0
        total_pruned = 0
        
        layer_stats = {}
        
        for name, mask in pruning_masks.items():
            layer_params = mask.numel()
            layer_pruned = (mask == 0).sum().item()
            layer_sparsity = layer_pruned / layer_params if layer_params > 0 else 0
            
            layer_stats[name] = {
                'total_params': layer_params,
                'pruned_params': layer_pruned,
                'sparsity': layer_sparsity
            }
            
            total_params += layer_params
            total_pruned += layer_pruned
        
        overall_sparsity = total_pruned / total_params if total_params > 0 else 0
        
        stats = {
            'overall_sparsity': overall_sparsity,
            'total_parameters': total_params,
            'pruned_parameters': total_pruned,
            'layer_stats': layer_stats
        }
        
        return stats


class BenchmarkSuite:
    """Complete benchmarking suite for comparing pruning methods"""
    
    def __init__(
        self, 
        model_name: str,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compare_pruning_methods(
        self,
        open_domain_texts: List[str],
        domain_texts: List[str],
        test_texts: List[str],
        sparsity_ratio: float = 0.5,
        batch_size: int = 4
    ) -> Dict[str, Dict[str, float]]:
        """Compare all pruning methods"""
        from d_pruner import DPruner, DPrunerConfig, DomainSpecificDataset  # Import from main module
        
        results = {}
        
        # Test each method
        methods = ['dense', 'magnitude', 'sparsegpt', 'llm_pruner', 'd_pruner']
        
        for method in methods:
            logger.info(f"Evaluating {method} method...")
            
            # Load fresh model for each method
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            evaluator = ModelEvaluator(model, self.tokenizer, self.device)
            
            if method == 'dense':
                # Baseline dense model
                perplexity = evaluator.evaluate_perplexity(test_texts, batch_size)
                results[method] = {
                    'perplexity': perplexity,
                    'sparsity': 0.0
                }
            
            elif method == 'magnitude':
                # Magnitude pruning
                pruner = MagnitudePruner(model)
                masks = pruner.prune(sparsity_ratio)
                
                perplexity = evaluator.evaluate_perplexity(test_texts, batch_size)
                stats = evaluator.compute_sparsity_stats(masks)
                
                results[method] = {
                    'perplexity': perplexity,
                    'sparsity': stats['overall_sparsity']
                }
            
            elif method == 'sparsegpt':
                # SparseGPT simulation
                dataset = DomainSpecificDataset(open_domain_texts + domain_texts, self.tokenizer)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                pruner = SparseGPTSimulator(model, self.device)
                masks = pruner.prune(dataloader, sparsity_ratio)
                
                perplexity = evaluator.evaluate_perplexity(test_texts, batch_size)
                stats = evaluator.compute_sparsity_stats(masks)
                
                results[method] = {
                    'perplexity': perplexity,
                    'sparsity': stats['overall_sparsity']
                }
            
            elif method == 'llm_pruner':
                # LLM-Pruner simulation
                dataset = DomainSpecificDataset(open_domain_texts + domain_texts, self.tokenizer)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                pruner = LLMPrunerSimulator(model, self.tokenizer, self.device)
                masks = pruner.prune(dataloader, sparsity_ratio)
                
                perplexity = evaluator.evaluate_perplexity(test_texts, batch_size)
                stats = evaluator.compute_sparsity_stats(masks)
                
                results[method] = {
                    'perplexity': perplexity,
                    'sparsity': stats['overall_sparsity']
                }
            
            elif method == 'd_pruner':
                # D-PRUNER
                config = DPrunerConfig(
                    model_name=self.model_name,
                    sparsity_ratio=sparsity_ratio,
                    batch_size=batch_size,
                    lambda_reg=0.1,
                    device=self.device
                )
                
                pruner = DPruner(
                    self.model_name,
                    self.device,
                    config.lambda_reg,
                    config.alpha
                )
                
                # Replace the model in pruner with our current model
                pruner.model = model
                
                # Run D-PRUNER steps
                pruner.step1_compute_general_importance(open_domain_texts, batch_size)
                pruner.step2_compute_dual_importance(domain_texts, batch_size, 1)
                pruner.step3_prune_model(sparsity_ratio, False, 128)
                
                perplexity = evaluator.evaluate_perplexity(test_texts, batch_size)
                stats = evaluator.compute_sparsity_stats(pruner.pruning_masks)
                
                results[method] = {
                    'perplexity': perplexity,
                    'sparsity': stats['overall_sparsity']
                }
            
            logger.info(f"{method} results: {results[method]}")
        
        return results
    
    def generate_comparison_report(
        self, 
        results: Dict[str, Dict[str, float]],
        output_file: Optional[str] = None
    ) -> str:
        """Generate a detailed comparison report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PRUNING METHODS COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append()
        
        # Summary table
        report_lines.append("SUMMARY TABLE")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Method':<15} {'Perplexity':<12} {'Sparsity':<10}")
        report_lines.append("-" * 40)
        
        for method, metrics in results.items():
            perplexity = metrics.get('perplexity', 0)
            sparsity = metrics.get('sparsity', 0)
            report_lines.append(f"{method:<15} {perplexity:<12.2f} {sparsity:<10.1%}")
        
        report_lines.append("-" * 40)
        report_lines.append()
        
        # Detailed analysis
        report_lines.append("DETAILED ANALYSIS")
        report_lines.append("-" * 40)
        
        if 'dense' in results:
            dense_perplexity = results['dense']['perplexity']
            
            for method, metrics in results.items():
                if method != 'dense':
                    perplexity = metrics.get('perplexity', 0)
                    sparsity = metrics.get('sparsity', 0)
                    perplexity_increase = ((perplexity - dense_perplexity) / dense_perplexity) * 100
                    
                    report_lines.append(f"{method.upper()}:")
                    report_lines.append(f"  Sparsity: {sparsity:.1%}")
                    report_lines.append(f"  Perplexity: {perplexity:.2f}")
                    report_lines.append(f"  Perplexity increase vs dense: {perplexity_increase:+.1f}%")
                    report_lines.append()
        
        # Performance ranking
        report_lines.append("PERFORMANCE RANKING (by perplexity)")
        report_lines.append("-" * 40)
        
        # Sort by perplexity (lower is better)
        sorted_methods = sorted(
            [(method, metrics) for method, metrics in results.items()],
            key=lambda x: x[1].get('perplexity', float('inf'))
        )
        
        for i, (method, metrics) in enumerate(sorted_methods, 1):
            perplexity = metrics.get('perplexity', 0)
            sparsity = metrics.get('sparsity', 0)
            report_lines.append(f"{i}. {method:<15} (PPL: {perplexity:.2f}, Sparsity: {sparsity:.1%})")
        
        report_lines.append()
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report


def run_comprehensive_evaluation(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    domain: str = "medical",
    sparsity_ratio: float = 0.5,
    output_dir: str = "./evaluation_results"
) -> Dict[str, Dict[str, float]]:
    """Run comprehensive evaluation of all pruning methods"""
    
    # Load sample data
    from d_pruner import load_sample_data
    open_domain_texts, domain_texts, test_texts = load_sample_data(domain)
    
    # Reduce dataset size for faster evaluation
    open_domain_texts = open_domain_texts[:50]
    domain_texts = domain_texts[:50]
    test_texts = test_texts[:20]
    
    # Initialize benchmark suite
    benchmark = BenchmarkSuite(model_name)
    
    # Run comparison
    results = benchmark.compare_pruning_methods(
        open_domain_texts=open_domain_texts,
        domain_texts=domain_texts,
        test_texts=test_texts,
        sparsity_ratio=sparsity_ratio,
        batch_size=2  # Small batch size for demo
    )
    
    # Generate report
    report = benchmark.generate_comparison_report(
        results, 
        f"{output_dir}/pruning_comparison_report.txt"
    )
    
    print(report)
    return results


if __name__ == "__main__":
    # Example usage
    results = run_comprehensive_evaluation(
        model_name="microsoft/DialoGPT-small",  # Using smaller model for demo
        domain="medical",
        sparsity_ratio=0.3,
        output_dir="./evaluation_results"
    )
    
    print("\nFinal Results:")
    for method, metrics in results.items():
        print(f"{method}: {metrics}")