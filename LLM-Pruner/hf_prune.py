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
    else:
        importance_summary = {}
    
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
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    )
    model.config.use_cache = False
    if args.device != "cpu":
        model.half()
    model.to(args.device)

    if args.test_before_train:
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
        logger.log("PPL before pruning: {}".format(ppl))

    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

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
    
    # Initialize importance scores storage
    importance_scores = {}
    
    if args.block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "channel_groups": {
            },
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            "round_to": model.config.num_attention_heads,   # ✅ add this

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
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            # Store importance scores before pruning step
            if hasattr(imp, 'importance_scores'):
                importance_scores.update(imp.importance_scores)

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
        
            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
         
         # --- FIX CONFIG AFTER PRUNING ---
        first_layer = model.model.layers[0]

        # update hidden size & intermediate size
        model.config.hidden_size = model.model.embed_tokens.embedding_dim
        model.config.intermediate_size = first_layer.mlp.gate_proj.out_features

        # keep head_dim constant (e.g. 64)
        head_dim = first_layer.self_attn.head_dim
        new_hidden = model.config.hidden_size

        # recompute attention heads so reshape works
        model.config.num_attention_heads = new_hidden // head_dim
        model.config.num_key_value_heads = model.config.num_attention_heads

        print(f"[Fix] hidden_size={new_hidden}, head_dim={head_dim}, "
              f"num_heads={model.config.num_attention_heads}")

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner

    elif args.channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers":[],
            #"round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                #LlamaAttention: llama_pruner.hf_attention_pruner,
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
                example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            # Store importance scores before pruning step
            if hasattr(imp, 'importance_scores'):
                importance_scores.update(imp.importance_scores)

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()
        
        del pruner
            
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError
    
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
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
        save_pruning_results(model, importance_scores, pruning_info, save_dir, logger)
    
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

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        # =================== FIX CONFIG AFTER PRUNING ===================
        model.config.hidden_size = model.model.embed_tokens.embedding_dim
        model.config.intermediate_size = model.model.layers[0].mlp.gate_proj.out_features

        # Compute num_attention_heads dynamically
        head_dim = model.model.layers[0].self_attn.head_dim
        model.config.num_attention_heads = model.config.hidden_size // head_dim
        model.config.num_key_value_heads = model.config.num_attention_heads

        assert model.config.hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"
        # ================================================================

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
        
        logger.log("\n==================Finish================\n")
    
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
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

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
