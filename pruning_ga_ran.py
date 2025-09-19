"""
LLM Pruning using Genetic Algorithm with Block-based Pruning and Importance Metrics
Author: Assistant
Description: Implements evolutionary structured pruning for Large Language Models with importance-aware fitness
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
import json
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
import pickle
import os
from huggingface_hub import HfApi, login

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm"""
    population_size: int = 3
    num_generations: int = 10
    mutation_rate: float = 0.1
    pruning_rate: float = 0.4
    block_size: int = 8
    num_parents: int = 2
    num_children: int = 3
    eval_samples: int = 300  # Changed default to 300
    seed: int = 42
    importance_lambda: float = 0.0  # Lambda hyperparameter for importance metric
    importance_dict_path: Optional[str] = None  # Path to importance dictionary

class BlockPruningMask:
    """Manages block-based pruning masks for model weights"""
   
    def __init__(self, shape: Tuple[int, int], block_size: int = 8, pruning_rate: float = 0.4):
        self.shape = shape
        self.block_size = block_size
        self.pruning_rate = pruning_rate
       
        # Calculate number of blocks
        self.n_blocks_h = (shape[0] + block_size - 1) // block_size
        self.n_blocks_w = (shape[1] + block_size - 1) // block_size
        self.total_blocks = self.n_blocks_h * self.n_blocks_w
       
        # Initialize block mask (1 = keep, 0 = prune)
        self.block_mask = np.ones((self.n_blocks_h, self.n_blocks_w), dtype=np.float32)
    
    def importance_aware_prune(self, importance_tensor=None):
      """Prune blocks based on importance scores with probabilistic selection"""
      n_prune = int(self.total_blocks * self.pruning_rate)

      if importance_tensor is None:
        # Fallback to random pruning
        self.random_prune()
        return
    
      # Calculate block-level importance by averaging weights in each block
      block_importance = np.zeros((self.n_blocks_h, self.n_blocks_w))

      for i in range(self.n_blocks_h):
        for j in range(self.n_blocks_w):
            start_h = i * self.block_size
            end_h = min(start_h + self.block_size, self.shape[0])
            start_w = j * self.block_size
            end_w = min(start_w + self.block_size, self.shape[1])
            
            # Average importance in this block
            block_values = importance_tensor[start_h:end_h, start_w:end_w]
            block_importance[i, j] = block_values.mean().item()
    
      # Convert importance to pruning probabilities
      # Lower importance = Higher pruning probability
      max_importance = block_importance.max()
      min_importance = block_importance.min()

      if max_importance == min_importance:
        # All blocks have same importance, use random
        prune_probs = np.full_like(block_importance, 0.5)
      else:
        # Normalize: low importance -> high prob, high importance -> low prob
        normalized = (block_importance - min_importance) / (max_importance - min_importance)
        prune_probs = 1.0 - normalized  # Invert: low importance gets high probability
    
      # Probabilistic selection of blocks to prune
      all_blocks = [(i, j, prune_probs[i, j]) for i in range(self.n_blocks_h) 
                  for j in range(self.n_blocks_w)]
    
      # Sort by probability (highest first) and select top n_prune
      all_blocks.sort(key=lambda x: x[2], reverse=True)

      self.block_mask = np.ones((self.n_blocks_h, self.n_blocks_w), dtype=np.float32)

      # Prune the most likely blocks
      for idx in range(min(n_prune, len(all_blocks))):
        i, j, prob = all_blocks[idx]
        # Add some randomness: prune with probability
        if random.random() < (prob * 0.8 + 0.2):  # Ensure some randomness
            self.block_mask[i, j] = 0
   
    def random_prune(self):
      """Random pruning method"""
      n_prune = int(self.total_blocks * self.pruning_rate)
      all_indices = list(range(self.total_blocks))
      random.shuffle(all_indices)
      indices_to_prune = all_indices[:n_prune]

      self.block_mask = np.ones((self.n_blocks_h, self.n_blocks_w), dtype=np.float32)
      for idx in indices_to_prune:
        i = idx // self.n_blocks_w
        j = idx % self.n_blocks_w
        self.block_mask[i, j] = 0

           
    def get_weight_mask(self) -> torch.Tensor:
        """Convert block mask to weight-level mask"""
        mask = torch.zeros(self.shape, dtype=torch.float32)
       
        for i in range(self.n_blocks_h):
            for j in range(self.n_blocks_w):
                start_h = i * self.block_size
                end_h = min(start_h + self.block_size, self.shape[0])
                start_w = j * self.block_size
                end_w = min(start_w + self.block_size, self.shape[1])
               
                mask[start_h:end_h, start_w:end_w] = float(self.block_mask[i, j])
               
        return mask
   
    def crossover(self, other: 'BlockPruningMask') -> 'BlockPruningMask':
        """Perform crossover with another mask"""
        child = BlockPruningMask(self.shape, self.block_size, self.pruning_rate)
       
        # Uniform crossover at block level
        for i in range(self.n_blocks_h):
            for j in range(self.n_blocks_w):
                if random.random() < 0.5:
                    child.block_mask[i, j] = self.block_mask[i, j]
                else:
                    child.block_mask[i, j] = other.block_mask[i, j]
       
        # Ensure pruning rate is maintained
        child.enforce_pruning_rate()
        return child
   
    def mutate(self, mutation_rate: float = 0.1):
        """Apply mutation to the mask"""
        n_mutate = max(1, int(self.total_blocks * mutation_rate))
       
        for _ in range(n_mutate):
            # Randomly select two blocks and swap their states
            idx1 = random.randint(0, self.total_blocks - 1)
            idx2 = random.randint(0, self.total_blocks - 1)
           
            i1, j1 = idx1 // self.n_blocks_w, idx1 % self.n_blocks_w
            i2, j2 = idx2 // self.n_blocks_w, idx2 % self.n_blocks_w
           
            # Swap only if it maintains different states
            if self.block_mask[i1, j1] != self.block_mask[i2, j2]:
                self.block_mask[i1, j1], self.block_mask[i2, j2] = \
                    self.block_mask[i2, j2], self.block_mask[i1, j1]
   
    def enforce_pruning_rate(self):
        """Adjust mask to ensure exact pruning rate"""
        current_pruned = np.sum(self.block_mask == 0)
        target_pruned = int(self.total_blocks * self.pruning_rate)
       
        if current_pruned < target_pruned:
            # Need to prune more
            keep_indices = np.where(self.block_mask.flatten() == 1)[0]
            to_prune = np.random.choice(keep_indices, target_pruned - current_pruned, replace=False)
            for idx in to_prune:
                i, j = idx // self.n_blocks_w, idx % self.n_blocks_w
                self.block_mask[i, j] = 0
               
        elif current_pruned > target_pruned:
            # Need to keep more
            prune_indices = np.where(self.block_mask.flatten() == 0)[0]
            to_keep = np.random.choice(prune_indices, current_pruned - target_pruned, replace=False)
            for idx in to_keep:
                i, j = idx // self.n_blocks_w, idx % self.n_blocks_w
                self.block_mask[i, j] = 1

class Individual:
    """Represents an individual in the genetic algorithm population"""
   
    def __init__(self, model_name: str, config: GAConfig):
        self.model_name = model_name
        self.config = config
        self.masks = {}
        self.fitness = float('inf')  # Lower is better
        self.perplexity = float('inf')
        self.importance_score = 0.0
        self.generation = 0
       
    def initialize_random_masks(self, model: nn.Module, importance_handler=None):
        """Initialize completely random pruning masks for prunable layers"""
        for name, module in model.named_modules():
            if self._is_prunable_layer(name, module):
                weight_shape = module.weight.shape
                mask = BlockPruningMask(weight_shape, self.config.block_size, self.config.pruning_rate)
                
                # Always use random pruning for initialization
                mask.random_prune()
                self.masks[name] = mask
               
    def _is_prunable_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer should be pruned"""
        # Prune only Linear layers in attention and FFN
        if not isinstance(module, nn.Linear):
            return False
           
        # Avoid pruning critical layers
        avoid_patterns = ['embed', 'lm_head', 'norm', 'layernorm', 'head']
        for pattern in avoid_patterns:
            if pattern in name.lower():
                return False
               
        # Focus on attention and FFN layers
        prune_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'gate_proj', 'up_proj', 'down_proj',
                         'fc1', 'fc2', 'mlp']
        for pattern in prune_patterns:
            if pattern in name.lower():
                return True
               
        return False
   
    def apply_masks(self, model: nn.Module):
      """Apply pruning masks to model weights with proper device handling"""
      with torch.no_grad():
        for name, module in model.named_modules():
            if name in self.masks:
                # Get mask and ensure it's on the right device and dtype
                mask = self.masks[name].get_weight_mask()
                mask = mask.to(device=module.weight.device, dtype=module.weight.dtype)
                
                # Apply mask
                module.weight.data *= mask
                
                # Optional: Set small weights to exactly zero for better sparsity
                # module.weight.data[torch.abs(module.weight.data) < 1e-6] = 0
                   
    def crossover(self, other: 'Individual') -> 'Individual':
        """Create offspring through crossover"""
        child = Individual(self.model_name, self.config)
        child.generation = max(self.generation, other.generation) + 1
       
        for layer_name in self.masks:
            if layer_name in other.masks:
                child.masks[layer_name] = self.masks[layer_name].crossover(other.masks[layer_name])
            else:
                child.masks[layer_name] = copy.deepcopy(self.masks[layer_name])
               
        return child
   
    def mutate(self):
        """Apply mutation to the individual"""
        for mask in self.masks.values():
            if random.random() < 0.5:  # Mutate each layer with 50% probability
                mask.mutate(self.config.mutation_rate)

class ImportanceMetricHandler:
    """Handles importance metric loading and matching with model layers"""
   
    def __init__(self, importance_dict_path: Optional[str] = None):
        self.importance_dict = {}
        self.layer_name_mapping = {}
       
        if importance_dict_path and os.path.exists(importance_dict_path):
            self.load_importance_dict(importance_dict_path)
   
    def load_importance_dict(self, path: str):
        """Load importance dictionary from pickle file"""
        try:
            with open(path, 'rb') as f:
                self.importance_dict = pickle.load(f)
            logger.info(f"Loaded importance dictionary with {len(self.importance_dict)} layers")
            logger.info(f"Sample keys: {list(self.importance_dict.keys())[:5]}")
        except Exception as e:
            logger.error(f"Failed to load importance dictionary: {e}")
            self.importance_dict = {}
   
    def match_layer_names(self, model_layer_name: str, importance_keys: List[str]) -> Optional[str]:
        """Match model layer name with importance dictionary keys"""
        # Try direct match first
        if model_layer_name in importance_keys:
            return model_layer_name
       
        # Try with 'model.' prefix
        with_prefix = f"model.{model_layer_name}"
        if with_prefix in importance_keys:
            return with_prefix
       
        # Try with '.weight' suffix
        with_suffix = f"{model_layer_name}.weight"
        if with_suffix in importance_keys:
            return with_suffix
       
        # Try with both prefix and suffix
        with_both = f"model.{model_layer_name}.weight"
        if with_both in importance_keys:
            return with_both
       
        # Try to match patterns
        for key in importance_keys:
            # Remove 'model.' prefix and '.weight' suffix for comparison
            clean_key = key.replace('model.', '').replace('.weight', '')
            if clean_key == model_layer_name:
                return key
           
            # Check if the model layer name is contained in the key
            if model_layer_name in key or key in model_layer_name:
                return key
       
        return None
   
    def calculate_kept_importance(self, masks: Dict[str, Any], model: nn.Module) -> float:
        """Calculate total importance of kept (non-pruned) weights"""
        if not self.importance_dict:
            return 0.0
       
        total_importance = 0.0
        importance_keys = list(self.importance_dict.keys())
       
        for layer_name, mask in masks.items():
            # Find matching importance key
            matched_key = self.match_layer_names(layer_name, importance_keys)
           
            if matched_key is None:
                logger.debug(f"No importance metric found for layer: {layer_name}")
                continue
           
            # Get importance tensor
            importance_tensor = self.importance_dict[matched_key]
           
            # Convert to tensor if needed
            if not isinstance(importance_tensor, torch.Tensor):
                importance_tensor = torch.tensor(importance_tensor, dtype=torch.float32)
            importance_tensor = importance_tensor.cpu().float()
            # Get weight mask
            weight_mask = mask.get_weight_mask()
           
            # Check shape compatibility
            if importance_tensor.shape != weight_mask.shape:
                logger.warning(f"Shape mismatch for {layer_name}: "
                             f"importance {importance_tensor.shape} vs mask {weight_mask.shape}")
                continue
           
            # Calculate importance of kept weights
            kept_importance = (importance_tensor * weight_mask).sum().item()
            total_importance += kept_importance
           
        return total_importance

class GeneticPruner:
    """Main class for genetic algorithm-based pruning"""
   
    def __init__(self, model_name: str, config: GAConfig):
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
       
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
           
        # Load evaluation dataset
        logger.info("Loading evaluation dataset")
        self.eval_dataloader = self._prepare_dataloader()
       
        # Initialize importance metric handler
        self.importance_handler = ImportanceMetricHandler(config.importance_dict_path)
       
        # Population
        self.population = []
        self.best_individual = None
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_perplexity': [],
            'best_importance': []
        }
       
    def _prepare_dataloader(self):
      """Prepare evaluation dataloader with samples from WikiText-2 dataset"""
      # Load WikiText-2 dataset
      dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
      
      # Filter out empty texts and very short texts
      def filter_text(examples):
        filtered_texts = []
        for text in examples['text']:
            # Remove empty lines and very short texts
            text = text.strip()
            if len(text) > 50:  # Only keep texts longer than 50 characters
                filtered_texts.append(text)
        return {'text': filtered_texts}
      
      # Filter and clean the dataset
      filtered_dataset = dataset.map(filter_text, batched=True, remove_columns=['text'])
      
      # Remove empty batches
      filtered_dataset = filtered_dataset.filter(lambda x: len(x['text']) > 0)
      
      def tokenize_function(examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
    
      tokenized = filtered_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
      tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

      # Sample subset for evaluation (default 300 samples)
      num_samples = min(self.config.eval_samples, len(tokenized))
      indices = random.sample(range(len(tokenized)), num_samples)
      subset = tokenized.select(indices)

      return torch.utils.data.DataLoader(subset, batch_size=4, shuffle=False)
   
    def evaluate_fitness(self, individual: Individual) -> float:
      """Evaluate fitness with perplexity and importance metric - with nan protection"""
      model_copy = copy.deepcopy(self.model)
      individual.apply_masks(model_copy)
      model_copy.eval()

      # Calculate perplexity with stability checks
      total_loss = 0
      total_tokens = 0
      valid_batches = 0

      with torch.no_grad():
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Shift for language modeling
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = model_copy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Check for valid loss
                if outputs.loss.isnan() or outputs.loss.isinf():
                    logger.warning("Invalid loss detected, skipping batch")
                    continue
                    
                batch_loss = outputs.loss.item()
                batch_tokens = attention_mask.sum().item()
                
                # Additional safety check
                if batch_tokens > 0 and not np.isnan(batch_loss) and not np.isinf(batch_loss):
                    total_loss += batch_loss * batch_tokens
                    total_tokens += batch_tokens
                    valid_batches += 1
                    
            except Exception as e:
                logger.warning(f"Error in batch evaluation: {e}")
                continue
    
      # Calculate perplexity with safety checks
      if total_tokens == 0 or valid_batches == 0:
        logger.error("No valid tokens found during evaluation")
        perplexity = 1000.0  # High penalty for invalid model
      else:
        avg_loss = total_loss / total_tokens
        # Clamp loss to prevent overflow
        avg_loss = np.clip(avg_loss, 0, 10)
        perplexity = np.exp(avg_loss)
        
        # Additional safety check for perplexity
        if np.isnan(perplexity) or np.isinf(perplexity):
            logger.warning("Invalid perplexity calculated, using penalty")
            perplexity = 1000.0
    
      individual.perplexity = perplexity
    
      # Calculate importance of kept weights (your existing code is fine here)
      try:
        importance_score = self.importance_handler.calculate_kept_importance(
            individual.masks, self.model
        )
        individual.importance_score = importance_score
      except Exception as e:
        logger.warning(f"Error calculating importance: {e}")
        importance_score = 0.0
        individual.importance_score = 0.0
    
      # Combined fitness: perplexity - lambda * importance
      fitness = perplexity - self.config.importance_lambda * importance_score
    
      # Safety check for fitness
      if np.isnan(fitness) or np.isinf(fitness):
        fitness = 1000.0
        
      individual.fitness = fitness
    
      logger.info(f"Valid batches: {valid_batches}/{len(self.eval_dataloader)}, "
               f"Perplexity: {perplexity:.2f}, Importance: {importance_score:.2f}, "
               f"Fitness: {fitness:.2f} (lambda={self.config.importance_lambda})")
    
      del model_copy
      torch.cuda.empty_cache()
    
      return fitness
   
    def initialize_population(self):
      """Initialize population with random pruning"""
      logger.info(f"Initializing population of size {self.config.population_size} with random pruning")

      for i in range(self.config.population_size):
        individual = Individual(self.model_name, self.config)
        individual.initialize_random_masks(self.model, self.importance_handler)
        self.population.append(individual)
    
    def quick_diversity_check(self):
      """Quick check if individuals are different"""
      if len(self.population) < 2:
        return
    
      # Check first layer of first two individuals
      first_layer = list(self.population[0].masks.keys())[0]
      mask1 = self.population[0].masks[first_layer].block_mask
      mask2 = self.population[1].masks[first_layer].block_mask

      identical = np.array_equal(mask1, mask2)
      logger.info(f"First two individuals identical: {identical}")
      logger.info(f"Individual 0 first layer pruned blocks: {np.sum(mask1 == 0)}")
      logger.info(f"Individual 1 first layer pruned blocks: {np.sum(mask2 == 0)}")
       
    def select_parents(self) -> List[Individual]:
        """Select best individuals as parents"""
        # Sort by fitness (lower is better)
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        return sorted_pop[:self.config.num_parents]
   
    def create_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring from parents"""
        offspring = []
       
        for _ in range(self.config.num_children):
            # Random parent selection for crossover
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            child.mutate()
            offspring.append(child)
           
        return offspring
   
    def evolve(self):
        """Main evolution loop"""
        logger.info("Starting evolution")
        logger.info(f"Using importance lambda: {self.config.importance_lambda}")
       
        # Initialize population
        self.initialize_population()
       
        for generation in range(self.config.num_generations):
            logger.info(f"\n=== Generation {generation + 1}/{self.config.num_generations} ===")
           
            # Evaluate fitness for new individuals
            for individual in self.population:
                if individual.fitness == float('inf'):
                    fitness = self.evaluate_fitness(individual)
           
            # Record statistics
            fitnesses = [ind.fitness for ind in self.population]
            perplexities = [ind.perplexity for ind in self.population]
            importances = [ind.importance_score for ind in self.population]
           
            best_fitness = min(fitnesses)
            avg_fitness = np.mean(fitnesses)
            best_idx = fitnesses.index(best_fitness)
           
            self.history['generation'].append(generation + 1)
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_perplexity'].append(perplexities[best_idx])
            self.history['best_importance'].append(importances[best_idx])
           
            logger.info(f"Best fitness: {best_fitness:.2f}, Average: {avg_fitness:.2f}")
            logger.info(f"Best perplexity: {perplexities[best_idx]:.2f}, "
                       f"Best importance: {importances[best_idx]:.2f}")
           
            # Update best individual
            best_current = min(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or best_current.fitness < self.best_individual.fitness:
                self.best_individual = copy.deepcopy(best_current)
           
            # Last generation - no need to create offspring
            if generation == self.config.num_generations - 1:
                break
           
            # Selection and reproduction
            parents = self.select_parents()
            offspring = self.create_offspring(parents)
           
            # Evaluate offspring
            for child in offspring:
                fitness = self.evaluate_fitness(child)
           
            # Survivor selection - keep best from parents and offspring
            all_individuals = parents + offspring
            all_individuals.sort(key=lambda x: x.fitness)
            self.population = all_individuals[:self.config.population_size]
           
        logger.info(f"\nEvolution complete!")
        logger.info(f"Best fitness: {self.best_individual.fitness:.2f}")
        logger.info(f"Best perplexity: {self.best_individual.perplexity:.2f}")
        logger.info(f"Best importance score: {self.best_individual.importance_score:.2f}")
       
    def save_results(self, output_dir: str = "pruning_results"):
        """Save pruning results and plots"""
        os.makedirs(output_dir, exist_ok=True)
       
        # Save best masks (both block and weight level)
        mask_data = {}
        for name, mask in self.best_individual.masks.items():
            mask_data[name] = {
                'block_mask': mask.block_mask.tolist(),
                'weight_mask': mask.get_weight_mask().numpy().tolist(),  # Add weight-level mask
                'shape': mask.shape,
                'block_size': mask.block_size,
                'pruning_rate': mask.pruning_rate
            }
       
        with open(f"{output_dir}/best_masks.json", 'w') as f:
            json.dump(mask_data, f, indent=2)
       
        # Save evolution history
        history_with_metrics = self.history.copy()
        history_with_metrics['lambda'] = self.config.importance_lambda
        history_with_metrics['best_final_fitness'] = self.best_individual.fitness
        history_with_metrics['best_final_perplexity'] = self.best_individual.perplexity
        history_with_metrics['best_final_importance'] = self.best_individual.importance_score
       
        with open(f"{output_dir}/evolution_history.json", 'w') as f:
            json.dump(history_with_metrics, f, indent=2)
       
        # Plot evolution progress
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
       
        # Plot 1: Fitness evolution
        axes[0, 0].plot(self.history['generation'], self.history['best_fitness'],
                       label='Best Fitness', marker='o')
        axes[0, 0].plot(self.history['generation'], self.history['avg_fitness'],
                       label='Average Fitness', marker='s')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title(f'Fitness Evolution (λ={self.config.importance_lambda})')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
       
        # Plot 2: Perplexity evolution
        axes[0, 1].plot(self.history['generation'], self.history['best_perplexity'],
                       label='Best Perplexity', marker='o', color='green')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
       
        # Plot 3: Importance evolution
        axes[1, 0].plot(self.history['generation'], self.history['best_importance'],
                       label='Best Importance Score', marker='o', color='red')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Importance Score')
        axes[1, 0].set_title('Importance Score Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
       
        # Plot 4: Combined metrics
        axes[1, 1].plot(self.history['generation'], self.history['best_perplexity'],
                       label='Perplexity', marker='o')
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.history['generation'], self.history['best_importance'],
                label='Importance', marker='s', color='red')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Perplexity', color='blue')
        ax2.set_ylabel('Importance Score', color='red')
        axes[1, 1].set_title('Perplexity vs Importance Trade-off')
        axes[1, 1].grid(True)
       
        plt.tight_layout()
        plt.savefig(f"{output_dir}/evolution_plots.png", dpi=150)
        plt.close()
       
        logger.info(f"Results saved to {output_dir}/")
       
    def apply_best_pruning(self, hf_repo_id: Optional[str] = None, local_save_path: Optional[str] = None):
      """Apply best pruning and upload to HuggingFace Hub"""
      if self.best_individual is None:
        raise ValueError("No best individual found. Run evolution first.")
    
      # Apply masks directly (no deepcopy to save memory)
      self.best_individual.apply_masks(self.model)

      # Create a temporary local directory if uploading to HF
      if hf_repo_id and not local_save_path:
        local_save_path = "./temp_pruned_model"
    
      if local_save_path:
        logger.info(f"Saving pruned model locally to {local_save_path}")
        os.makedirs(local_save_path, exist_ok=True)
        
        # Save with memory-efficient sharding
        self.model.save_pretrained(local_save_path, max_shard_size="1GB")
        self.tokenizer.save_pretrained(local_save_path)
        
        # Add model card with pruning info
        self._create_model_card(local_save_path)
    
      # Upload to Hugging Face Hub
      if hf_repo_id:
        try:
            logger.info(f"Uploading pruned model to HuggingFace Hub: {hf_repo_id}")
            
            # Push to hub
            self.model.push_to_hub(
                hf_repo_id,
                commit_message=f"Pruned model - {self.config.pruning_rate:.1%} pruning rate, perplexity: {self.best_individual.perplexity:.2f}"
            )
            self.tokenizer.push_to_hub(hf_repo_id)
            
            logger.info(f"Successfully uploaded to: https://huggingface.co/{hf_repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace Hub: {e}")
            logger.info("Make sure you're logged in: huggingface-cli login")
            
        finally:
            # Cleanup temp directory if created
            if local_save_path == "./temp_pruned_model":
                import shutil
                shutil.rmtree(local_save_path, ignore_errors=True)
    
      return self.model
    
    def _create_model_card(self, save_path: str):
      """Create a model card with pruning information"""
      model_card_content = f"""---
language: en
license: apache-2.0
base_model: {self.model_name}
tags:
- pruned
- genetic-algorithm
- block-pruning
---

# Pruned {self.model_name}

This model has been pruned using a genetic algorithm with block-based structured pruning.

## Pruning Details
- **Pruning Rate**: {self.config.pruning_rate:.1%}
- **Block Size**: {self.config.block_size}x{self.config.block_size}
- **Final Perplexity**: {self.best_individual.perplexity:.2f}
- **Importance Score**: {self.best_individual.importance_score:.2f}
- **Population Size**: {self.config.population_size}
- **Generations**: {self.config.num_generations}

## Performance
The pruned model maintains reasonable performance while reducing model size through structured pruning of attention and MLP layers.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{save_path.split('/')[-1]}")
tokenizer = AutoTokenizer.from_pretrained("{save_path.split('/')[-1]}")
```
"""

      with open(f"{save_path}/README.md", 'w') as f:
        f.write(model_card_content)

         

def main():
    parser = argparse.ArgumentParser(description='LLM Pruning with Genetic Algorithm and Importance Metrics')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Model to prune')
    parser.add_argument('--generations', type=int, default=10,
                       help='Number of generations')
    parser.add_argument('--population', type=int, default=3,
                       help='Population size')
    parser.add_argument('--hf-repo', type=str, default=None,
                       help='HuggingFace Hub repository ID (e.g., username/model-name)')
    parser.add_argument('--pruning-rate', type=float, default=0.4,
                       help='Target pruning rate')
    parser.add_argument('--block-size', type=int, default=8,
                       help='Block size for pruning')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Mutation rate')
    parser.add_argument('--eval-samples', type=int, default=300,
                       help='Number of samples for evaluation (default: 300)')
    parser.add_argument('--importance-dict', type=str, default=None,
                       help='Path to importance dictionary pickle file')
    parser.add_argument('--importance-lambda', type=float, default=0.0,
                       help='Lambda hyperparameter for importance metric (default: 0.0)')
    parser.add_argument('--output-dir', type=str, default='pruning_results',
                       help='Output directory for results')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save pruned model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
   
    args = parser.parse_args()
   
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
   
    # Create configuration
    config = GAConfig(
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        pruning_rate=args.pruning_rate,
        block_size=args.block_size,
        eval_samples=args.eval_samples,
        importance_lambda=args.importance_lambda,
        importance_dict_path=args.importance_dict,
        seed=args.seed
    )
   
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Generations: {config.num_generations}")
    logger.info(f"  Population: {config.population_size}")
    logger.info(f"  Pruning rate: {config.pruning_rate}")
    logger.info(f"  Block size: {config.block_size}")
    logger.info(f"  Mutation rate: {config.mutation_rate}")
    logger.info(f"  Eval samples: {config.eval_samples}")
    logger.info(f"  Importance lambda: {config.importance_lambda}")
    logger.info(f"  Importance dict: {config.importance_dict_path}")
   
    # Run genetic pruning
    pruner = GeneticPruner(args.model, config)
    pruner.evolve()
    pruner.save_results(args.output_dir)
   
    # Update the main function call:
    if args.save_model or args.hf_repo:
      pruner.apply_best_pruning(
      hf_repo_id=args.hf_repo,
      local_save_path=args.save_model
      )
   
    logger.info("Pruning complete!")

if __name__ == "__main__":
    main()
