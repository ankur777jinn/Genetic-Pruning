"""
D-PRUNER: Domain-specific LLM Extractor
Implementation of the dual-pruning methodology for domain-specific compression of LLMs.

Paper: "Pruning as a Domain-specific LLM Extractor"
arXiv: https://arxiv.org/pdf/2405.06275
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import math
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainSpecificDataset(Dataset):
    """Dataset for domain-specific calibration data"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone()
        }


class GeneralImportanceCalculator:
    """Calculate general weight importance using open-domain calibration data"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        
    def compute_general_importance(
        self, 
        dataloader: DataLoader,
        target_layers: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute general weight importance using Taylor series approximation
        
        Args:
            dataloader: DataLoader with open-domain calibration data
            target_layers: List of layer names to compute importance for
            
        Returns:
            Dict mapping layer names to importance scores
        """
        logger.info("Computing general weight importance...")
        
        if target_layers is None:
            target_layers = self._get_target_layers()
            
        self.model.eval()
        importance_scores = {}
        
        # Initialize gradients accumulator
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in target_layers):
                param.grad = torch.zeros_like(param)
        
        total_loss = 0
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Computing general importance"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item()
            num_samples += batch["input_ids"].size(0)
        
        # Compute importance scores using Taylor series approximation
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in target_layers) and param.grad is not None:
                # Importance = |∂L/∂W * W + 0.5 * W * H_mm * W|
                # We approximate H_mm using squared gradients (Fisher information)
                grad_squared = param.grad ** 2
                importance = torch.abs(param.grad * param + 0.5 * param * grad_squared * param)
                importance_scores[name] = importance.detach().clone()
        
        # Zero out gradients
        self.model.zero_grad()
        
        logger.info(f"Computed importance for {len(importance_scores)} layers")
        return importance_scores
    
    def _get_target_layers(self) -> List[str]:
        """Get default target layers for pruning"""
        target_layers = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Self-attention projections
            "gate_proj", "up_proj", "down_proj"      # MLP projections
        ]
        return target_layers


class DualPruningLoss:
    """Implements the regularized loss function for dual pruning"""
    
    def __init__(
        self, 
        general_importance: Dict[str, torch.Tensor],
        lambda_reg: float = 0.1,
        alpha: float = 1e-4
    ):
        self.general_importance = general_importance
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        
    def compute_regularized_loss(
        self, 
        model: nn.Module,
        original_weights: Dict[str, torch.Tensor],
        next_token_loss: torch.Tensor,
        gradients: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the regularized loss with general importance term
        
        Args:
            model: The model being trained
            original_weights: Original weight values before update
            next_token_loss: Standard next token prediction loss
            gradients: Current gradients for each parameter
            
        Returns:
            Total regularized loss
        """
        regularization_loss = 0
        
        for name, param in model.named_parameters():
            if name in self.general_importance and name in gradients:
                G_m = self.general_importance[name]  # General importance
                g_m = gradients[name]  # Current gradient
                
                # Regularization term: λ * α² * G_m * (g_m)²
                reg_term = self.lambda_reg * (self.alpha ** 2) * G_m * (g_m ** 2)
                regularization_loss += reg_term.sum()
        
        total_loss = next_token_loss + regularization_loss
        return total_loss


class DPruner:
    """Main D-PRUNER implementation for domain-specific LLM compression"""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        lambda_reg: float = 0.1,
        alpha: float = 1e-4
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.general_importance = None
        self.dual_importance = None
        
    def step1_compute_general_importance(
        self, 
        open_domain_texts: List[str],
        batch_size: int = 8
    ):
        """Step 1: Compute general weight importance"""
        logger.info("Step 1: Computing general weight importance")
        
        # Create dataset and dataloader
        dataset = DomainSpecificDataset(open_domain_texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Compute general importance
        importance_calculator = GeneralImportanceCalculator(self.model, self.device)
        self.general_importance = importance_calculator.compute_general_importance(dataloader)
        
        logger.info("Step 1 completed: General importance computed")
    
    def step2_compute_dual_importance(
        self,
        domain_texts: List[str],
        batch_size: int = 8,
        num_epochs: int = 1
    ):
        """Step 2: Compute dual importance scores using domain-specific data"""
        logger.info("Step 2: Computing dual importance scores")
        
        if self.general_importance is None:
            raise ValueError("Must compute general importance first (step 1)")
        
        # Store original weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            if name in self.general_importance:
                original_weights[name] = param.data.clone()
        
        # Create dataset and dataloader
        dataset = DomainSpecificDataset(domain_texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize dual pruning loss
        dual_loss = DualPruningLoss(
            self.general_importance, 
            self.lambda_reg, 
            self.alpha
        )
        
        self.model.train()
        dual_importance_scores = {}
        
        # Initialize importance accumulators
        for name in self.general_importance:
            dual_importance_scores[name] = torch.zeros_like(
                self.general_importance[name]
            )
        
        num_samples = 0
        
        for epoch in range(num_epochs):
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                next_token_loss = outputs.loss
                
                # Compute gradients
                next_token_loss.backward(retain_graph=True)
                
                # Collect gradients
                gradients = {}
                for name, param in self.model.named_parameters():
                    if name in self.general_importance and param.grad is not None:
                        gradients[name] = param.grad.clone()
                
                # Compute regularized loss gradients
                self.model.zero_grad()
                total_loss = dual_loss.compute_regularized_loss(
                    self.model, original_weights, next_token_loss, gradients
                )
                total_loss.backward()
                
                # Accumulate dual importance scores using empirical Fisher
                for name, param in self.model.named_parameters():
                    if name in self.general_importance and param.grad is not None:
                        # Dual importance using Taylor approximation with regularized loss
                        importance = torch.abs(
                            param.grad * param + 0.5 * (param.grad * param) ** 2
                        )
                        dual_importance_scores[name] += importance.detach()
                
                num_samples += batch["input_ids"].size(0)
                self.model.zero_grad()
        
        # Average importance scores
        for name in dual_importance_scores:
            dual_importance_scores[name] /= num_samples
        
        self.dual_importance = dual_importance_scores
        logger.info("Step 2 completed: Dual importance computed")
    
    def step3_prune_model(
        self, 
        sparsity_ratio: float = 0.5,
        use_iterative_blocking: bool = False,
        block_size: int = 128
    ):
        """Step 3: Prune the model based on dual importance scores"""
        logger.info(f"Step 3: Pruning model with {sparsity_ratio*100}% sparsity")
        
        if self.dual_importance is None:
            raise ValueError("Must compute dual importance first (step 2)")
        
        pruning_masks = {}
        
        for name, param in self.model.named_parameters():
            if name in self.dual_importance:
                importance = self.dual_importance[name]
                
                if use_iterative_blocking:
                    mask = self._compute_iterative_blocking_mask(
                        importance, sparsity_ratio, block_size
                    )
                else:
                    mask = self._compute_layerwise_mask(importance, sparsity_ratio)
                
                pruning_masks[name] = mask
                
                # Apply pruning mask
                param.data *= mask
        
        self.pruning_masks = pruning_masks
        logger.info("Step 3 completed: Model pruned")
        
        # Log pruning statistics
        self._log_pruning_stats()
    
    def _compute_layerwise_mask(
        self, 
        importance: torch.Tensor, 
        sparsity_ratio: float
    ) -> torch.Tensor:
        """Compute pruning mask layer-wise"""
        flat_importance = importance.flatten()
        num_params = flat_importance.numel()
        num_to_prune = int(num_params * sparsity_ratio)
        
        # Get threshold for pruning
        threshold = torch.topk(flat_importance, num_to_prune, largest=False).values.max()
        mask = (importance > threshold).float()
        
        return mask
    
    def _compute_iterative_blocking_mask(
        self, 
        importance: torch.Tensor, 
        sparsity_ratio: float, 
        block_size: int
    ) -> torch.Tensor:
        """Compute pruning mask with iterative blocking"""
        if importance.dim() == 1:
            return self._compute_layerwise_mask(importance, sparsity_ratio)
        
        # For 2D weight matrices
        rows, cols = importance.shape
        mask = torch.ones_like(importance)
        
        for start_col in range(0, cols, block_size):
            end_col = min(start_col + block_size, cols)
            block_importance = importance[:, start_col:end_col]
            
            flat_importance = block_importance.flatten()
            num_params = flat_importance.numel()
            num_to_prune = int(num_params * sparsity_ratio)
            
            if num_to_prune > 0:
                threshold = torch.topk(flat_importance, num_to_prune, largest=False).values.max()
                block_mask = (block_importance > threshold).float()
                mask[:, start_col:end_col] = block_mask
        
        return mask
    
    def _log_pruning_stats(self):
        """Log pruning statistics"""
        total_params = 0
        pruned_params = 0
        
        for name, param in self.model.named_parameters():
            if name in self.pruning_masks:
                mask = self.pruning_masks[name]
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()
        
        sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruning completed: {sparsity:.2%} sparsity achieved")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Pruned parameters: {pruned_params:,}")
    
    def save_pruned_model(self, output_dir: str):
        """Save the pruned model and masks"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save pruning masks
        if hasattr(self, 'pruning_masks'):
            mask_path = output_path / "pruning_masks.pt"
            torch.save(self.pruning_masks, mask_path)
        
        # Save importance scores
        if self.general_importance is not None:
            importance_path = output_path / "general_importance.pt"
            torch.save(self.general_importance, importance_path)
        
        if self.dual_importance is not None:
            importance_path = output_path / "dual_importance.pt"
            torch.save(self.dual_importance, importance_path)
        
        logger.info(f"Pruned model saved to {output_path}")
    
    def load_pruned_model(self, model_dir: str):
        """Load a previously pruned model"""
        model_path = Path(model_dir)
        
        # Load masks
        mask_path = model_path / "pruning_masks.pt"
        if mask_path.exists():
            self.pruning_masks = torch.load(mask_path)
            
            # Apply masks to model
            for name, param in self.model.named_parameters():
                if name in self.pruning_masks:
                    param.data *= self.pruning_masks[name]
        
        logger.info(f"Loaded pruned model from {model_path}")
    
    def evaluate_perplexity(self, test_texts: List[str], batch_size: int = 8) -> float:
        """Evaluate model perplexity on test texts"""
        dataset = DomainSpecificDataset(test_texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating perplexity"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        perplexity = math.exp(avg_loss)
        return perplexity


class DPrunerConfig:
    """Configuration class for D-PRUNER"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        lambda_reg: float = 0.1,
        alpha: float = 1e-4,
        sparsity_ratio: float = 0.5,
        use_iterative_blocking: bool = False,
        block_size: int = 128,
        batch_size: int = 8,
        num_epochs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.sparsity_ratio = sparsity_ratio
        self.use_iterative_blocking = use_iterative_blocking
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device


def run_dpruner_pipeline(
    config: DPrunerConfig,
    open_domain_texts: List[str],
    domain_specific_texts: List[str],
    output_dir: str,
    test_texts: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Run the complete D-PRUNER pipeline
    
    Args:
        config: D-PRUNER configuration
        open_domain_texts: Open-domain calibration texts
        domain_specific_texts: Domain-specific calibration texts
        output_dir: Directory to save pruned model
        test_texts: Optional test texts for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Starting D-PRUNER pipeline...")
    
    # Initialize D-PRUNER
    pruner = DPruner(
        config.model_name,
        config.device,
        config.lambda_reg,
        config.alpha
    )
    
    # Step 1: Compute general importance
    pruner.step1_compute_general_importance(
        open_domain_texts, 
        config.batch_size
    )
    
    # Step 2: Compute dual importance
    pruner.step2_compute_dual_importance(
        domain_specific_texts,
        config.batch_size,
        config.num_epochs
    )
    
    # Step 3: Prune model
    pruner.step3_prune_model(
        config.sparsity_ratio,
        config.use_iterative_blocking,
        config.block_size
    )
    
    # Save pruned model
    pruner.save_pruned_model(output_dir)
    
    # Evaluate if test texts provided
    results = {}
    if test_texts:
        perplexity = pruner.evaluate_perplexity(test_texts, config.batch_size)
        results["perplexity"] = perplexity
        logger.info(f"Test perplexity: {perplexity:.2f}")
    
    logger.info("D-PRUNER pipeline completed!")
    return results


# Example usage and utility functions
def load_sample_data(domain: str = "medical") -> Tuple[List[str], List[str], List[str]]:
    """Load sample data for different domains"""
    
    if domain == "medical":
        # Sample medical texts (in practice, load from actual datasets)
        open_domain = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Climate change affects weather patterns globally."
        ] * 100
        
        domain_specific = [
            "The patient presented with acute myocardial infarction symptoms.",
            "Diagnosis confirmed through electrocardiogram and cardiac biomarkers.",
            "Treatment protocol includes antiplatelet therapy and beta-blockers."
        ] * 100
        
        test_texts = [
            "Medical diagnosis requires careful evaluation of patient symptoms.",
            "Pharmaceutical interventions must consider drug interactions.",
            "Clinical trials demonstrate treatment efficacy and safety."
        ] * 50
        
    elif domain == "legal":
        open_domain = [
            "The weather is nice today with sunny skies.",
            "Technology advances continue to shape our future.",
            "Education plays a vital role in society."
        ] * 100
        
        domain_specific = [
            "The plaintiff filed a motion for summary judgment.",
            "Contract law governs the formation and enforcement of agreements.",
            "Constitutional rights protect individual liberties under due process."
        ] * 100
        
        test_texts = [
            "Legal precedent guides judicial decision-making processes.",
            "Statutory interpretation requires careful analysis of legislative intent.",
            "Civil procedure rules ensure fair and orderly litigation."
        ] * 50
    
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    return open_domain, domain_specific, test_texts


if __name__ == "__main__":
    # Example usage
    config = DPrunerConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        lambda_reg=0.1,
        sparsity_ratio=0.5,
        batch_size=4,
        num_epochs=1
    )
    
    # Load sample data
    open_domain_texts, domain_texts, test_texts = load_sample_data("medical")
    
    # Run D-PRUNER pipeline
    results = run_dpruner_pipeline(
        config=config,
        open_domain_texts=open_domain_texts,
        domain_specific_texts=domain_texts,
        output_dir="./pruned_model",
        test_texts=test_texts
    )
    
    print("Results:", results)
