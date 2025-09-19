import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple, Dict, Any
import pickle

import json
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from transformers import AutoModelForCausalLM, AutoTokenizer
import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def capture_importance_scores_taylor(model, args, logger):
    """
    Capture importance scores using Taylor expansion method
    """
    logger.log("Computing Taylor importance scores...")
    importance_scores = {}
    
    # Get examples for computing gradients
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len=64).to(args.device)
    
    model.train()  # Enable gradient computation
    model.zero_grad()
    
    if args.taylor in ['param_mix', 'param_second']:
        # For second-order methods, accumulate gradients
        for j in range(args.num_examples):
            batch_input = example_prompts[j].unsqueeze(0)
            loss = model(batch_input, labels=batch_input).loss
            loss.backward()
            
            # Accumulate squared gradients for second-order Taylor
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_squared = param.grad.pow(2) / args.num_examples
                    if hasattr(param, 'acc_grad'):
                        param.acc_grad += grad_squared
                    else:
                        param.acc_grad = grad_squared.clone()
            
            model.zero_grad()
        
        # Final forward pass
        loss = model(example_prompts, labels=example_prompts).loss
        loss.backward()
        
        # Compute importance scores
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if args.taylor == 'param_second':
                    # Second-order Taylor: param^2 * accumulated_grad^2
                    if hasattr(param, 'acc_grad'):
                        score = param.pow(2) * param.acc_grad
                    else:
                        score = param.pow(2) * param.grad.pow(2)
                elif args.taylor == 'param_mix':
                    # Mixed method
                    if hasattr(param, 'acc_grad'):
                        score = torch.abs(param) * param.acc_grad
                    else:
                        score = torch.abs(param * param.grad)
                else:
                    # Default first-order
                    score = torch.abs(param * param.grad)
                
                importance_scores[name] = score.detach().clone()
    
    else:  # param_first or vectorize
        # Standard first-order Taylor
        loss = model(example_prompts, labels=example_prompts).loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if args.taylor == 'param_first':
                    # First-order Taylor: |param * grad|
                    score = torch.abs(param * param.grad)
                else:
                    # Vectorize or default: |grad|
                    score = torch.abs(param.grad)
                
                importance_scores[name] = score.detach().clone()
    
    logger.log(f"Captured {len(importance_scores)} importance score tensors")
    return importance_scores

def save_pruning_results(model, importance_scores, pruning_info, save_path, logger):
    """
    Save pruned model weights and importance matrix in readable formats
    
    Args:
        model: The pruned model
        importance_scores: Dictionary containing importance scores for different layers
        pruning_info: Dictionary containing pruning configuration and statistics
        save_path: Base path for saving files
        logger: Logger instance for logging
    """
    logger.log("Saving pruning results...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Save model weights in organized format
    weights_info = {}
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Convert to numpy for easier inspection
            weight_np = param.data.cpu().numpy()
            
            weights_info[name] = {
                'shape': weight_np.shape,
                'dtype': str(weight_np.dtype),
                'mean': float(np.mean(weight_np)),
                'std': float(np.std(weight_np)),
                'min': float(np.min(weight_np)),
                'max': float(np.max(weight_np)),
                'num_params': int(np.prod(weight_np.shape)),
                'sparsity': float(np.sum(weight_np == 0) / weight_np.size) if weight_np.size > 0 else 0.0
            }
            
            # Save actual weights
            weight_file = os.path.join(save_path, f"{name.replace('.', '_')}_weights.npy")
            np.save(weight_file, weight_np)
            weights_info[name]['weight_file'] = weight_file
    
    # 2. Save importance scores if available
    if importance_scores:
        importance_file = os.path.join(save_path, "importance_scores.pkl")
        with open(importance_file, 'wb') as f:
            pickle.dump(importance_scores, f)
        
        # Create readable summary of importance scores
        importance_summary = {}
        for layer_name, scores in importance_scores.items():
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
                importance_summary[layer_name] = {
                    'shape': scores_np.shape,
                    'mean': float(np.mean(scores_np)),
                    'std': float(np.std(scores_np)),
                    'min': float(np.min(scores_np)),
                    'max': float(np.max(scores_np)),
                    'top_10_indices': np.argsort(scores_np.flatten())[-10:].tolist(),
                    'bottom_10_indices': np.argsort(scores_np.flatten())[:10].tolist()
                }
                
                # Save full importance scores as numpy array
                imp_file = os.path.join(save_path, f"{layer_name.replace('.', '_')}_importance.npy")
                np.save(imp_file, scores_np)
                importance_summary[layer_name]['importance_file'] = imp_file
                
        logger.log(f"Saved {len(importance_scores)} importance score files")
    else:
        importance_summary = {}
        logger.log("No importance scores to save")
    
    # 3. Collect layer-wise statistics
    for i, layer in enumerate(model.model.layers):
        layer_name = f"layer_{i}"
        layer_stats[layer_name] = {
            'attention_heads': getattr(layer.self_attn, 'num_heads', 'N/A'),
            'hidden_size': layer.self_attn.q_proj.weight.shape[1] if hasattr(layer.self_attn, 'q_proj') else 'N/A',
            'intermediate_size': layer.mlp.gate_proj.out_features if hasattr(layer.mlp, 'gate_proj') else 'N/A',
            'head_dim': getattr(layer.self_attn, 'head_dim', 'N/A')
        }
    
    # 4. Create comprehensive summary
    summary = {
        'model_info': {
            'model_name': getattr(model.config, 'name_or_path', 'Unknown'),
            'num_layers': len(model.model.layers),
            'hidden_size': model.config.hidden_size,
            'intermediate_size': getattr(model.config, 'intermediate_size', 'N/A'),
            'num_attention_heads': model.config.num_attention_heads,
            'vocab_size': model.config.vocab_size
        },
        'pruning_info': pruning_info,
        'layer_statistics': layer_stats,
        'weight_statistics': weights_info,
        'importance_statistics': importance_summary,
        'total_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'memory_usage_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }
    
    # Save summary as JSON
    summary_file = os.path.join(save_path, "pruning_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save summary as readable text
    text_summary_file = os.path.join(save_path, "pruning_summary.txt")
    with open(text_summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRUNING RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in summary['model_info'].items():
            f.write(f"{key:<25}: {value}\n")
        
        f.write(f"\nPRUNING INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in summary['pruning_info'].items():
            f.write(f"{key:<25}: {value}\n")
        
        f.write(f"\nLAYER STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for layer_name, stats in layer_stats.items():
            f.write(f"\n{layer_name}:\n")
            for key, value in stats.items():
                f.write(f"  {key:<20}: {value}\n")
        
        f.write(f"\nWEIGHT STATISTICS (Top 10 layers by parameter count):\n")
        f.write("-" * 40 + "\n")
        # Sort layers by parameter count
        sorted_weights = sorted(weights_info.items(), key=lambda x: x[1]['num_params'], reverse=True)
        for name, stats in sorted_weights[:10]:
            f.write(f"\n{name}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Parameters: {stats['num_params']:,}\n")
            f.write(f"  Sparsity: {stats['sparsity']:.4f}\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std: {stats['std']:.6f}\n")
    
    logger.log(f"Pruning results saved to: {save_path}")
    logger.log(f"- Summary: {summary_file}")
    logger.log(f"- Text summary: {text_summary_file}")
    logger.log(f"- Individual weight files: {len(weights_info)} files")
    if importance_scores:
        logger.log(f"- Importance scores: {len(importance_scores)} files")
    
    return summary_file

def extract_and_save_importance_scores(model, pruner, importance_scores, save_path, logger):
    """
    Extract detailed importance scores and save for visualization
    
    Args:
        model: The pruned model
        pruner: The pruner object (contains dependency information)
        importance_scores: Dictionary of importance scores
        save_path: Directory to save visualization data
        logger: Logger instance
    """
    logger.log("Extracting detailed importance scores for visualization...")
    
    # Initialize data structure
    visualization_data = {
        'layers': {},
        'summary': {
            'total_neurons': 0,
            'pruned_neurons': 0,
            'compression_ratio': 0
        },
        'metadata': {
            'model_name': getattr(model.config, 'name_or_path', 'Unknown'),
            'pruning_method': 'taylor',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    total_original_params = 0
    total_current_params = 0
    
    # Process each layer's importance scores
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            # Get original parameter shape info
            original_shape = param.shape
            current_shape = param.shape
            
            layer_info = {
                'layer_name': name,
                'layer_type': get_layer_type(name),
                'original_shape': list(original_shape),
                'current_shape': list(current_shape),
                'original_size': int(np.prod(original_shape)),
                'current_size': int(np.prod(current_shape)),
                'importance_scores': [],
                'pruned_indices': [],
                'statistics': {}
            }
            
            # Get importance scores if available
            if name in importance_scores:
                scores = importance_scores[name]
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.cpu().numpy().flatten()
                    layer_info['importance_scores'] = scores_np.tolist()
                    
                    # Calculate statistics
                    layer_info['statistics'] = {
                        'mean': float(np.mean(scores_np)),
                        'std': float(np.std(scores_np)),
                        'min': float(np.min(scores_np)),
                        'max': float(np.max(scores_np)),
                        'median': float(np.median(scores_np)),
                        'percentile_25': float(np.percentile(scores_np, 25)),
                        'percentile_75': float(np.percentile(scores_np, 75))
                    }
            
            # Identify pruned parameters (zeros in the weight matrix)
            if len(param.shape) == 2:  # Linear layers
                # For matrix, check if entire rows/columns are pruned
                weight_np = param.data.cpu().numpy()
                
                # Check for pruned output features (rows)
                pruned_outputs = np.where(np.all(weight_np == 0, axis=1))[0].tolist()
                
                # Check for pruned input features (columns)
                pruned_inputs = np.where(np.all(weight_np == 0, axis=0))[0].tolist()
                
                layer_info['pruned_indices'] = {
                    'output_features': pruned_outputs,
                    'input_features': pruned_inputs,
                    'total_pruned': len(pruned_outputs) + len(pruned_inputs)
                }
            else:
                # For other parameter types, just check for zero values
                weight_np = param.data.cpu().numpy()
                pruned_indices = np.where(weight_np == 0)[0].tolist() if weight_np.ndim == 1 else []
                layer_info['pruned_indices'] = pruned_indices
            
            visualization_data['layers'][name] = layer_info
            
            total_original_params += layer_info['original_size']
            total_current_params += layer_info['current_size']
    
    # Update summary statistics
    visualization_data['summary'] = {
        'total_neurons': total_original_params,
        'current_neurons': total_current_params,
        'pruned_neurons': total_original_params - total_current_params,
        'compression_ratio': total_current_params / total_original_params if total_original_params > 0 else 0,
        'pruning_percentage': (1 - total_current_params / total_original_params) * 100 if total_original_params > 0 else 0
    }
    
    # Save visualization data
    viz_file = os.path.join(save_path, 'importance_visualization_data.json')
    with open(viz_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    logger.log(f"Visualization data saved to: {viz_file}")
    
    # Create some basic plots
    create_basic_plots(visualization_data, save_path, logger)
    
    return viz_file

# Add this right after the pruning loop (around line 380)
def fix_model_config_after_pruning(model):
    """Fix model configuration after pruning"""
    # Get actual dimensions from the pruned model
    embed_dim = model.model.embed_tokens.weight.shape[1]
    
    # Update config
    model.config.hidden_size = embed_dim
    
    # Get attention dimensions from first layer
    first_attn = model.model.layers[0].self_attn
    head_dim = first_attn.head_dim
    
    # Calculate actual number of heads based on pruned q_proj
    actual_heads = first_attn.q_proj.weight.shape[0] // head_dim
    
    model.config.num_attention_heads = actual_heads
    model.config.num_key_value_heads = actual_heads
    
    # Update intermediate size
    model.config.intermediate_size = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    
    return model

# Call this function right after pruning
#model = fix_model_config_after_pruning(model)

def get_layer_type(layer_name):
    """Determine the type of layer from its name"""
    if 'self_attn' in layer_name:
        if 'q_proj' in layer_name:
            return 'attention_query'
        elif 'k_proj' in layer_name:
            return 'attention_key'
        elif 'v_proj' in layer_name:
            return 'attention_value'
        elif 'o_proj' in layer_name:
            return 'attention_output'
        else:
            return 'attention'
    elif 'mlp' in layer_name:
        if 'gate_proj' in layer_name:
            return 'mlp_gate'
        elif 'up_proj' in layer_name:
            return 'mlp_up'
        elif 'down_proj' in layer_name:
            return 'mlp_down'
        else:
            return 'mlp'
    elif 'embed_tokens' in layer_name:
        return 'embedding'
    elif 'norm' in layer_name:
        return 'normalization'
    else:
        return 'other'

'''def create_basic_plots(visualization_data, save_path, logger):
    """Create basic matplotlib visualizations"""
    logger.log("Creating basic visualization plots...")
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # 1. Layer-wise importance distribution
    layer_stats = []
    for name, layer_data in visualization_data['layers'].items():
        if layer_data['importance_scores']:
            stats = layer_data['statistics']
            layer_stats.append({
                'layer': name.split('.')[-2] + '.' + name.split('.')[-1],  # Shorter names
                'mean_importance': stats['mean'],
                'max_importance': stats['max'],
                'pruning_ratio': layer_data.get('pruned_indices', {}).get('total_pruned', 0) / layer_data['original_size'] if layer_data['original_size'] > 0 else 0
            })
    
    if layer_stats:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Mean importance by layer
        layers = [s['layer'] for s in layer_stats]
        means = [s['mean_importance'] for s in layer_stats]
        maxes = [s['max_importance'] for s in layer_stats]
        
        ax1.bar(range(len(layers)), means, alpha=0.7, label='Mean Importance')
        ax1.bar(range(len(layers)), maxes, alpha=0.5, label='Max Importance')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Importance Scores by Layer')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Pruning ratio by layer
        pruning_ratios = [s['pruning_ratio'] * 100 for s in layer_stats]
        ax2.bar(range(len(layers)), pruning_ratios, color='red', alpha=0.7)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Pruning Ratio (%)')
        ax2.set_title('Pruning Ratio by Layer')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'layer_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Overall importance distribution histogram
    all_scores = []
    for layer_data in visualization_data['layers'].values():
        if layer_data['importance_scores']:
            all_scores.extend(layer_data['importance_scores'])
    
    if all_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Importance Score')
        plt.ylabel('Frequency')
        plt.title('Overall Distribution of Importance Scores')
        plt.axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.6f}')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'importance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.log("Basic plots saved to PNG files")
'''
def modified_save_pruning_results(model, pruner, importance_scores, pruning_info, save_path, logger):
    """
    Enhanced version that includes importance score extraction
    """
    # Call original save function
    original_summary_file = save_pruning_results(model, importance_scores, pruning_info, save_path, logger)
    
    # Extract detailed importance scores for visualization
    viz_file = extract_and_save_importance_scores(model, pruner, importance_scores, save_path, logger)
    
    return original_summary_file, viz_file

def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args._dict,  # Fixed: was args._dict
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >= 1.9 else False  # Fixed: was >=1.9
    )
    model.config.use_cache = False
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    '''if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
    
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log("PPL before pruning: {}".format(ppl))'''

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device)

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    # Initialize importance scores storage - MOVED OUTSIDE CONDITIONAL BLOCKS
    importance_scores = {}
    
    # CAPTURE IMPORTANCE SCORES BEFORE PRUNING (for Taylor method)
    if pruner_type == 'taylor':
        logger.log("Capturing importance scores before pruning...")
        importance_scores = capture_importance_scores_taylor(model, args, logger)
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers":[],
            "channel_groups": {},
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "round_to": model.config.num_attention_heads,

            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None,
            "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                              [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):
            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len=64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            if module_param.grad is not None:
                                module_param.grad = module_param.grad * module_param.grad / args.num_examples
                                if hasattr(module_param, 'acc_grad'):
                                    module_param.acc_grad += module_param.grad
                                else:
                                    module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            # Perform pruning step
            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

            # modify inference-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        
        # Fix config after pruning
        first_layer = model.model.layers[0]
        model.config.hidden_size = model.model.embed_tokens.embedding_dim
        model.config.intermediate_size = first_layer.mlp.gate_proj.out_features

        head_dim = first_layer.self_attn.head_dim
        new_hidden = model.config.hidden_size
        model.config.num_attention_heads = new_hidden // head_dim
        model.config.num_key_value_heads = model.config.num_attention_heads

        print(f"[Fix] hidden_size={new_hidden}, head_dim={head_dim}, "
              f"num_heads={model.config.num_attention_heads}")

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None
        model = fix_model_config_after_pruning(model)

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio,
            "ignored_layers":[],
            "channel_groups": {},
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )
        model.zero_grad()
        
        logger.log("Start Pruning")
        for i in range(args.iterative_steps):
            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()
            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inference-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters, 100.0*after_pruning_parameters/before_pruning_parameters))
    
    # Prepare pruning information for saving
    pruning_info = {
        'base_model': args.base_model,
        'pruning_ratio': args.pruning_ratio,
        'pruner_type': pruner_type,
        'parameters_before': before_pruning_parameters,
        'parameters_after': after_pruning_parameters,
        'compression_ratio': 100.0 * after_pruning_parameters / before_pruning_parameters,
        'pruning_method': 'block_wise' if args.block_wise else ('channel_wise' if args.channel_wise else 'layer_wise'),
        'block_attention_layers': f"{args.block_attention_layer_start}-{args.block_attention_layer_end}" if args.block_wise else None,
        'block_mlp_layers': f"{args.block_mlp_layer_start}-{args.block_mlp_layer_end}" if args.block_wise else None,
        'iterative_steps': args.iterative_steps,
        'global_pruning': args.global_pruning,
        'taylor_method': args.taylor if pruner_type == 'taylor' else None,
        'num_examples': args.num_examples if pruner_type == 'taylor' else None,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save pruning results if requested
    if args.save_weights_importance:
        save_dir = os.path.join('prune_results', args.save_ckpt_log_name)
        if 'pruner' in locals() and pruner is not None:
            summary_file, viz_file = modified_save_pruning_results(
                model, pruner, importance_scores, pruning_info, save_dir, logger
            )
            logger.log(f"Visualization data ready! Load {viz_file} into the web visualizer.")
            # Clean up pruner after use
            del pruner
        else:
            # Fallback to original function if no pruner available
            save_pruning_results(model, importance_scores, pruning_info, save_dir, logger)
            logger.log("Pruner not available - saved basic results only.")
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
            'pruning_info': pruning_info
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    '''if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        # Fix config after pruning
        model.config.hidden_size = model.model.embed_tokens.embedding_dim
        model.config.intermediate_size = model.model.layers[0].mlp.gate_proj.out_features

        # Compute num_attention_heads dynamically
        head_dim = model.model.layers[0].self_attn.head_dim
        model.config.num_attention_heads = model.config.hidden_size // head_dim
        model.config.num_key_value_heads = model.config.num_attention_heads

        assert model.config.hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                
                result = tokenizer.decode(generation_output[0])
                logger.log(result)
        
        logger.log("\n==================Finish================\n")'''
        
        #ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
        #logger.log("PPL after pruning: {}".format(ppl))
        #logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if _name_ == "_main":  # FIXED: was if _name == "main":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}{pruner_type}{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    
    # NEW ARGUMENT FOR SAVING WEIGHTS AND IMPORTANCE
    parser.add_argument('--save_weights_importance', action='store_true', help='save pruned weights and importance matrix in readable format')
    
    args = parser.parse_args()

    torch_version = float('.'.join(torch._version_.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
