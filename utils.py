"""
Utility functions and data processing for D-PRUNER
"""

import torch
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import re
from datasets import load_dataset
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading utilities for different domains"""
    
    @staticmethod
    def load_medical_data() -> Tuple[List[str], List[str], List[str]]:
        """Load medical domain data"""
        
        # Open domain texts (general knowledge)
        open_domain = [
            "The weather today is sunny with clear skies.",
            "Technology continues to advance at a rapid pace.",
            "Education is fundamental to human development.",
            "Art and culture enrich our lives in many ways.",
            "Sports bring people together from different backgrounds.",
        ] * 50
        
        # Medical domain texts
        medical_texts = [
            "The patient presented with acute chest pain and shortness of breath.",
            "Diagnosis of myocardial infarction was confirmed by elevated troponin levels.",
            "Treatment includes antiplatelet therapy and beta-blocker administration.",
            "Surgical intervention may be required for severe coronary artery disease.",
            "Patient monitoring includes continuous ECG and vital sign assessment.",
            "Pharmacological management involves ACE inhibitors and statins.",
            "Risk factors include hypertension, diabetes, and smoking history.",
            "Prognosis depends on timely intervention and patient compliance.",
            "Follow-up care includes cardiac rehabilitation and lifestyle modifications.",
            "Complications may include arrhythmias and heart failure development."
        ] * 30
        
        # Test texts for evaluation
        test_texts = [
            "Medical diagnosis requires comprehensive patient evaluation and diagnostic testing.",
            "Clinical decision-making involves integration of symptoms, signs, and test results.",
            "Treatment protocols must consider patient-specific factors and contraindications.",
            "Healthcare quality depends on evidence-based medicine and continuous improvement.",
            "Patient safety is paramount in all medical interventions and procedures."
        ] * 20
        
        return open_domain, medical_texts, test_texts
    
    @staticmethod
    def load_legal_data() -> Tuple[List[str], List[str], List[str]]:
        """Load legal domain data"""
        
        # Open domain texts
        open_domain = [
            "The morning coffee was particularly good today.",
            "Music has the power to move people emotionally.",
            "Travel broadens perspectives and creates memories.",
            "Reading books expands knowledge and imagination.",
            "Exercise promotes both physical and mental health.",
        ] * 50
        
        # Legal domain texts
        legal_texts = [
            "The plaintiff filed a motion for summary judgment in federal court.",
            "Contract formation requires offer, acceptance, and valid consideration.",
            "Constitutional law governs the relationship between state and federal power.",
            "Criminal procedure ensures due process and protection of defendant rights.",
            "Tort law provides remedies for civil wrongs and personal injuries.",
            "Property law defines ownership rights and real estate transactions.",
            "Corporate governance involves fiduciary duties and shareholder rights.",
            "Administrative law regulates government agency actions and procedures.",
            "Evidence rules determine admissibility in judicial proceedings.",
            "Appellate review examines lower court decisions for legal errors."
        ] * 30
        
        # Test texts
        test_texts = [
            "Legal precedent guides judicial decision-making in similar cases.",
            "Statutory interpretation requires analysis of legislative intent and purpose.",
            "Civil litigation follows established procedural rules and deadlines.",
            "Legal research involves case law analysis and statutory construction.",
            "Professional responsibility governs attorney conduct and client relations."
        ] * 20
        
        return open_domain, legal_texts, test_texts
    
    @staticmethod
    def load_from_huggingface(dataset_name: str, split: str = "train", max_samples: int = 1000) -> List[str]:
        """Load data from Hugging Face datasets"""
        try:
            dataset = load_dataset(dataset_name, split=split)
            texts = []
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                # Extract text based on common field names
                text = ""
                if 'text' in item:
                    text = item['text']
                elif 'content' in item:
                    text = item['content']
                elif 'article' in item:
                    text = item['article']
                elif 'document' in item:
                    text = item['document']
                
                if text and len(text.strip()) > 50:  # Filter short texts
                    texts.append(text.strip())
            
            logger.info(f"Loaded {len(texts)} texts from {dataset_name}")
            return texts
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            return []
    
    @staticmethod
    def load_c4_data(max_samples: int = 1000) -> List[str]:
        """Load C4 dataset for open-domain calibration"""
        try:
            dataset = load_dataset("c4", "en", split="train", streaming=True)
            texts = []
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                text = item.get('text', '')
                if len(text) > 200:  # Filter short texts
                    # Clean and truncate text
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 2000:
                        text = text[:2000]
                    texts.append(text)
            
            logger.info(f"Loaded {len(texts)} C4 texts")
            return texts
            
        except Exception as e:
            logger.warning(f"Failed to load C4 data: {e}")
            return DataLoader.load_medical_data()[0]  # Fallback


class MaskAnalyzer:
    """Analyze pruning masks and weight distributions"""
    
    @staticmethod
    def analyze_mask_similarity(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """Compute similarity between two pruning masks"""
        if mask1.shape != mask2.shape:
            return 0.0
        
        # Count shared unpruned weights
        shared_ones = ((mask1 == 1) & (mask2 == 1)).sum().float()
        total_ones = max(mask1.sum().float(), mask2.sum().float())
        
        if total_ones == 0:
            return 1.0 if mask1.sum() == 0 and mask2.sum() == 0 else 0.0
        
        return (shared_ones / total_ones).item()
    
    @staticmethod
    def analyze_layer_sparsity(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze sparsity distribution across layers"""
        layer_sparsity = {}
        
        for name, mask in masks.items():
            total_params = mask.numel()
            pruned_params = (mask == 0).sum().item()
            sparsity = pruned_params / total_params if total_params > 0 else 0
            layer_sparsity[name] = sparsity
        
        return layer_sparsity
    
    @staticmethod
    def compare_domain_masks(
        healthcare_masks: Dict[str, torch.Tensor],
        legal_masks: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compare masks between different domains"""
        similarities = {}
        
        for name in healthcare_masks:
            if name in legal_masks:
                similarity = MaskAnalyzer.analyze_mask_similarity(
                    healthcare_masks[name],
                    legal_masks[name]
                )
                similarities[name] = similarity
        
        return similarities
    
    @staticmethod
    def visualize_mask_distribution(masks: Dict[str, torch.Tensor], save_path: Optional[str] = None):
        """Visualize mask distribution across layers"""
        try:
            import matplotlib.pyplot as plt
            
            layer_names = list(masks.keys())
            sparsity_values = []
            
            for name in layer_names:
                mask = masks[name]
                sparsity = (mask == 0).sum().item() / mask.numel()
                sparsity_values.append(sparsity)
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(layer_names)), sparsity_values)
            plt.xlabel('Layer Index')
            plt.ylabel('Sparsity Ratio')
            plt.title('Sparsity Distribution Across Layers')
            plt.xticks(range(len(layer_names)), [name.split('.')[-1] for name in layer_names], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Sparsity visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")


class PerformanceTracker:
    """Track and analyze model performance during pruning"""
    
    def __init__(self):
        self.metrics_history = []
        self.sparsity_levels = []
    
    def add_measurement(
        self, 
        sparsity: float, 
        perplexity: float, 
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
        method: str = "unknown"
    ):
        """Add a performance measurement"""
        measurement = {
            'sparsity': sparsity,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'method': method
        }
        self.metrics_history.append(measurement)
        self.sparsity_levels.append(sparsity)
    
    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """Get Pareto frontier of sparsity vs performance trade-offs"""
        if not self.metrics_history:
            return []
        
        # Sort by sparsity
        sorted_metrics = sorted(self.metrics_history, key=lambda x: x['sparsity'])
        
        pareto_frontier = []
        best_perplexity = float('inf')
        
        for metric in sorted_metrics:
            if metric['perplexity'] < best_perplexity:
                best_perplexity = metric['perplexity']
                pareto_frontier.append(metric)
        
        return pareto_frontier
    
    def export_metrics(self, filepath: str):
        """Export metrics to CSV file"""
        if not self.metrics_history:
            logger.warning("No metrics to export")
            return
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics exported to {filepath}")
    
    def plot_performance_curves(self, save_path: Optional[str] = None):
        """Plot performance curves"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                logger.warning("No metrics to plot")
                return
            
            # Group by method
            methods = {}
            for metric in self.metrics_history:
                method = metric['method']
                if method not in methods:
                    methods[method] = {'sparsity': [], 'perplexity': []}
                methods[method]['sparsity'].append(metric['sparsity'])
                methods[method]['perplexity'].append(metric['perplexity'])
            
            plt.figure(figsize=(10, 6))
            for method, data in methods.items():
                plt.plot(data['sparsity'], data['perplexity'], 'o-', label=method)
            
            plt.xlabel('Sparsity Ratio')
            plt.ylabel('Perplexity')
            plt.title('Performance vs Sparsity Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Performance curves saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class ConfigManager:
    """Manage D-PRUNER configurations"""
    
    DEFAULT_CONFIG = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "lambda_reg": 0.1,
        "alpha": 1e-4,
        "sparsity_ratio": 0.5,
        "use_iterative_blocking": False,
        "block_size": 128,
        "batch_size": 8,
        "num_epochs": 1,
        "device": "cuda",
        "max_length": 512,
        "calibration_samples": 1000,
        "evaluation_samples": 100
    }
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with default config
            final_config = ConfigManager.DEFAULT_CONFIG.copy()
            final_config.update(config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return final_config
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return ConfigManager.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    @staticmethod
    def create_domain_configs() -> Dict[str, Dict[str, Any]]:
        """Create domain-specific configurations"""
        base_config = ConfigManager.DEFAULT_CONFIG.copy()
        
        configs = {
            "healthcare": {
                **base_config,
                "lambda_reg": 0.1,
                "alpha": 3e-4,
                "domain": "healthcare"
            },
            "legal": {
                **base_config,
                "lambda_reg": 0.001,
                "alpha": 3e-4,
                "domain": "legal"
            },
            "finance": {
                **base_config,
                "lambda_reg": 0.05,
                "alpha": 2e-4,
                "domain": "finance"
            }
        }
        
        return configs


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk_words = words[i:i + max_length]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def extract_domain_keywords(texts: List[str], domain: str) -> List[str]:
        """Extract domain-specific keywords"""
        domain_keywords = {
            "healthcare": [
                "patient", "diagnosis", "treatment", "medical", "clinical",
                "symptoms", "therapy", "medication", "hospital", "doctor",
                "disease", "condition", "healthcare", "pharmaceutical", "surgery"
            ],
            "legal": [
                "court", "judge", "lawyer", "legal", "law", "case", "trial",
                "contract", "agreement", "litigation", "statute", "regulation",
                "plaintiff", "defendant", "jurisdiction", "precedent", "ruling"
            ],
            "finance": [
                "investment", "portfolio", "market", "stock", "bond", "asset",
                "risk", "return", "capital", "financial", "trading", "banking",
                "credit", "debt", "equity", "dividend", "profit", "loss"
            ]
        }
        
        return domain_keywords.get(domain, [])
    
    @staticmethod
    def compute_domain_score(text: str, domain: str) -> float:
        """Compute domain relevance score for text"""
        keywords = TextProcessor.extract_domain_keywords([], domain)
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Normalize by text length and keyword list length
        words_in_text = len(text.split())
        if words_in_text == 0 or len(keywords) == 0:
            return 0.0
        
        score = keyword_count / min(words_in_text, len(keywords))
        return min(score, 1.0)  # Cap at 1.0


class ExperimentRunner:
    """Run comprehensive experiments with different configurations"""
    
    def __init__(self, base_output_dir: str = "./experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_tracker = PerformanceTracker()
    
    def run_sparsity_sweep(
        self,
        model_name: str,
        open_domain_texts: List[str],
        domain_texts: List[str],
        test_texts: List[str],
        sparsity_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        domain: str = "medical"
    ) -> Dict[float, Dict[str, float]]:
        """Run experiments across different sparsity levels"""
        
        from d_pruner import DPruner, DPrunerConfig
        
        results = {}
        
        for sparsity in sparsity_levels:
            logger.info(f"Running experiment with {sparsity*100}% sparsity")
            
            # Create experiment directory
            exp_dir = self.base_output_dir / f"{domain}_sparsity_{sparsity:.1f}"
            exp_dir.mkdir(exist_ok=True)
            
            try:
                # Configure D-PRUNER
                config = DPrunerConfig(
                    model_name=model_name,
                    sparsity_ratio=sparsity,
                    lambda_reg=0.1 if domain == "healthcare" else 0.001,
                    batch_size=4,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                # Run D-PRUNER
                pruner = DPruner(config.model_name, config.device, config.lambda_reg, config.alpha)
                
                # Execute pruning pipeline
                pruner.step1_compute_general_importance(open_domain_texts[:100], config.batch_size)
                pruner.step2_compute_dual_importance(domain_texts[:100], config.batch_size, 1)
                pruner.step3_prune_model(sparsity, False, 128)
                
                # Evaluate
                perplexity = pruner.evaluate_perplexity(test_texts[:20], config.batch_size)
                
                # Save results
                result = {
                    'perplexity': perplexity,
                    'sparsity': sparsity,
                    'config': config.__dict__
                }
                
                results[sparsity] = result
                
                # Save model and config
                pruner.save_pruned_model(str(exp_dir / "model"))
                ConfigManager.save_config(config.__dict__, str(exp_dir / "config.json"))
                
                # Track performance
                self.performance_tracker.add_measurement(
                    sparsity=sparsity,
                    perplexity=perplexity,
                    method="D-PRUNER"
                )
                
                logger.info(f"Sparsity {sparsity:.1f}: Perplexity = {perplexity:.2f}")
                
            except Exception as e:
                logger.error(f"Failed experiment with sparsity {sparsity}: {e}")
                results[sparsity] = {'error': str(e)}
        
        # Save overall results
        results_path = self.base_output_dir / f"{domain}_sparsity_sweep_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export performance tracking
        self.performance_tracker.export_metrics(
            str(self.base_output_dir / f"{domain}_performance_metrics.csv")
        )
        
        return results
    
    def run_hyperparameter_sweep(
        self,
        model_name: str,
        open_domain_texts: List[str],
        domain_texts: List[str],
        test_texts: List[str],
        lambda_values: List[float] = [0.01, 0.1, 1.0],
        alpha_values: List[float] = [1e-5, 1e-4, 1e-3],
        domain: str = "medical"
    ) -> Dict[str, Dict[str, float]]:
        """Run hyperparameter sweep"""
        
        from d_pruner import DPruner
        
        results = {}
        
        for lambda_reg in lambda_values:
            for alpha in alpha_values:
                experiment_name = f"lambda_{lambda_reg}_alpha_{alpha}"
                logger.info(f"Running experiment: {experiment_name}")
                
                try:
                    # Create pruner with specific hyperparameters
                    pruner = DPruner(model_name, "cuda", lambda_reg, alpha)
                    
                    # Run pruning pipeline
                    pruner.step1_compute_general_importance(open_domain_texts[:50], 4)
                    pruner.step2_compute_dual_importance(domain_texts[:50], 4, 1)
                    pruner.step3_prune_model(0.5, False, 128)
                    
                    # Evaluate
                    perplexity = pruner.evaluate_perplexity(test_texts[:10], 4)
                    
                    results[experiment_name] = {
                        'lambda_reg': lambda_reg,
                        'alpha': alpha,
                        'perplexity': perplexity,
                        'sparsity': 0.5
                    }
                    
                    logger.info(f"{experiment_name}: Perplexity = {perplexity:.2f}")
                    
                except Exception as e:
                    logger.error(f"Failed experiment {experiment_name}: {e}")
                    results[experiment_name] = {'error': str(e)}
        
        # Save results
        results_path = self.base_output_dir / f"{domain}_hyperparameter_sweep.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


class ModelSizeEstimator:
    """Estimate model size and memory requirements"""
    
    @staticmethod
    def estimate_model_size(model: torch.nn.Module) -> Dict[str, float]:
        """Estimate model size in MB"""
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        # Assume 4 bytes per parameter (float32)
        size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb': size_mb,
            'size_gb': size_mb / 1024
        }
    
    @staticmethod
    def estimate_pruned_size(
        original_size: Dict[str, float],
        sparsity_ratio: float
    ) -> Dict[str, float]:
        """Estimate size after pruning"""
        remaining_ratio = 1.0 - sparsity_ratio
        
        return {
            'total_parameters': int(original_size['total_parameters'] * remaining_ratio),
            'trainable_parameters': int(original_size['trainable_parameters'] * remaining_ratio),
            'size_mb': original_size['size_mb'] * remaining_ratio,
            'size_gb': original_size['size_gb'] * remaining_ratio,
            'compression_ratio': 1.0 / remaining_ratio if remaining_ratio > 0 else float('inf'),
            'space_saved_mb': original_size['size_mb'] * sparsity_ratio
        }
    
    @staticmethod
    def estimate_inference_speedup(sparsity_ratio: float) -> Dict[str, float]:
        """Estimate theoretical inference speedup"""
        # This is a rough approximation - actual speedup depends on hardware and sparsity pattern
        theoretical_speedup = 1.0 / (1.0 - sparsity_ratio) if sparsity_ratio < 1.0 else float('inf')
        
        # Practical speedup is usually lower due to overhead
        practical_speedup = min(theoretical_speedup * 0.7, 5.0)  # Cap at 5x
        
        return {
            'theoretical_speedup': theoretical_speedup,
            'practical_speedup': practical_speedup,
            'memory_reduction': sparsity_ratio
        }


def create_sample_experiment():
    """Create a sample experiment configuration"""
    
    # Load sample data
    from d_pruner import load_sample_data
    open_domain, domain_texts, test_texts = load_sample_data("medical")
    
    # Create experiment runner
    runner = ExperimentRunner("./sample_experiment")
    
    # Run sparsity sweep
    results = runner.run_sparsity_sweep(
        model_name="microsoft/DialoGPT-small",  # Small model for demo
        open_domain_texts=open_domain[:100],
        domain_texts=domain_texts[:100],
        test_texts=test_texts[:50],
        sparsity_levels=[0.1, 0.3, 0.5],
        domain="medical"
    )
    
    return results


if __name__ == "__main__":
    # Example usage of utilities
    
    # 1. Load and analyze data
    loader = DataLoader()
    open_domain, medical_texts, test_texts = loader.load_medical_data()
    
    print(f"Loaded {len(open_domain)} open domain texts")
    print(f"Loaded {len(medical_texts)} medical texts")
    print(f"Loaded {len(test_texts)} test texts")
    
    # 2. Analyze domain relevance
    processor = TextProcessor()
    sample_text = medical_texts[0]
    medical_score = processor.compute_domain_score(sample_text, "healthcare")
    legal_score = processor.compute_domain_score(sample_text, "legal")
    
    print(f"Sample text medical score: {medical_score:.2f}")
    print(f"Sample text legal score: {legal_score:.2f}")
    
    # 3. Create and save domain configurations
    configs = ConfigManager.create_domain_configs()
    for domain, config in configs.items():
        ConfigManager.save_config(config, f"./{domain}_config.json")
    
    print("Domain configurations created and saved")
    
    # 4. Demonstrate performance tracking
    tracker = PerformanceTracker()
    tracker.add_measurement(0.0, 5.2, 0.85, 0.82, "dense")
    tracker.add_measurement(0.3, 6.1, 0.82, 0.79, "d_pruner")
    tracker.add_measurement(0.5, 7.8, 0.78, 0.75, "d_pruner")
    
    pareto_frontier = tracker.get_pareto_frontier()
    print(f"Pareto frontier has {len(pareto_frontier)} points")
    
    print("Utilities demonstration completed!")