#!/usr/bin/env python3
"""
D-PRUNER Example Usage and Demo Script

This script demonstrates how to use D-PRUNER for domain-specific LLM compression
with comprehensive examples and comparisons with baseline methods.

Usage:
    python dpruner_example.py --domain medical --model microsoft/DialoGPT-small
    python dpruner_example.py --domain legal --sparsity 0.3 --compare-baselines
"""

import argparse
import logging
import torch
from pathlib import Path
import json
from typing import Dict, List, Any
import time

# Import D-PRUNER modules
from d_pruner import DPruner, DPrunerConfig, run_dpruner_pipeline, load_sample_data
from dpruner_evaluation import BenchmarkSuite, ModelEvaluator
from dpruner_utils import (
    DataLoader, ConfigManager, ExperimentRunner, 
    PerformanceTracker, MaskAnalyzer, ModelSizeEstimator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DPrunerDemo:
    """Complete D-PRUNER demonstration class"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        logger.info(f"D-PRUNER Demo initialized with model: {args.model}")
        logger.info(f"Domain: {args.domain}, Sparsity: {args.sparsity}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_basic_demo(self) -> Dict[str, Any]:
        """Run basic D-PRUNER demonstration"""
        logger.info("=== Running Basic D-PRUNER Demo ===")
        
        # Load data
        logger.info("Loading sample data...")
        open_domain_texts, domain_texts, test_texts = load_sample_data(self.args.domain)
        
        # Reduce dataset size for demo
        if self.args.quick_demo:
            open_domain_texts = open_domain_texts[:50]
            domain_texts = domain_texts[:50]
            test_texts = test_texts[:20]
        
        logger.info(f"Loaded {len(open_domain_texts)} open-domain texts")
        logger.info(f"Loaded {len(domain_texts)} domain-specific texts")
        logger.info(f"Loaded {len(test_texts)} test texts")
        
        # Configure D-PRUNER
        config = DPrunerConfig(
            model_name=self.args.model,
            sparsity_ratio=self.args.sparsity,
            lambda_reg=0.1 if self.args.domain == "healthcare" else 0.001,
            batch_size=self.args.batch_size,
            device=self.args.device
        )
        
        # Save configuration
        config_path = self.output_dir / "config.json"
        ConfigManager.save_config(config.__dict__, str(config_path))
        
        # Run D-PRUNER pipeline
        start_time = time.time()
        
        results = run_dpruner_pipeline(
            config=config,
            open_domain_texts=open_domain_texts,
            domain_specific_texts=domain_texts,
            output_dir=str(self.output_dir / "pruned_model"),
            test_texts=test_texts
        )
        
        end_time = time.time()
        pruning_time = end_time - start_time
        
        # Add timing information
        results['pruning_time_seconds'] = pruning_time
        results['sparsity_ratio'] = self.args.sparsity
        
        # Track performance
        self.performance_tracker.add_measurement(
            sparsity=self.args.sparsity,
            perplexity=results.get('perplexity', float('inf')),
            method="D-PRUNER"
        )
        
        logger.info(f"D-PRUNER completed in {pruning_time:.2f} seconds")
        logger.info(f"Final perplexity: {results.get('perplexity', 'N/A')}")
        
        return results
    
    def run_comparison_demo(self) -> Dict[str, Dict[str, float]]:
        """Run comparison with baseline methods"""
        logger.info("=== Running Baseline Comparison Demo ===")
        
        # Load data
        open_domain_texts, domain_texts, test_texts = load_sample_data(self.args.domain)
        
        # Reduce for faster comparison
        open_domain_texts = open_domain_texts[:30]
        domain_texts = domain_texts[:30]
        test_texts = test_texts[:15]
        
        # Initialize benchmark suite
        benchmark = BenchmarkSuite(self.args.model)
        
        # Run comparison
        start_time = time.time()
        
        results = benchmark.compare_pruning_methods(
            open_domain_texts=open_domain_texts,
            domain_texts=domain_texts,
            test_texts=test_texts,
            sparsity_ratio=self.args.sparsity,
            batch_size=2  # Small batch for demo
        )
        
        end_time = time.time()
        comparison_time = end_time - start_time
        
        # Generate and save report
        report = benchmark.generate_comparison_report(
            results,
            str(self.output_dir / "comparison_report.txt")
        )
        
        # Save detailed results
        results_path = self.output_dir / "comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Track all methods
        for method, metrics in results.items():
            self.performance_tracker.add_measurement(
                sparsity=metrics.get('sparsity', 0),
                perplexity=metrics.get('perplexity', float('inf')),
                method=method
            )
        
        logger.info(f"Comparison completed in {comparison_time:.2f} seconds")
        print("\n" + "="*80)
        print("COMPARISON RESULTS SUMMARY")
        print("="*80)
        print(report)
        
        return results
    
    def run_sparsity_analysis(self) -> Dict[float, Dict[str, float]]:
        """Run analysis across different sparsity levels"""
        logger.info("=== Running Sparsity Analysis ===")
        
        # Load data
        open_domain_texts, domain_texts, test_texts = load_sample_data(self.args.domain)
        
        # Reduce dataset size
        open_domain_texts = open_domain_texts[:40]
        domain_texts = domain_texts[:40]
        test_texts = test_texts[:15]
        
        # Initialize experiment runner
        runner = ExperimentRunner(str(self.output_dir / "sparsity_experiments"))
        
        # Run sparsity sweep
        sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5] if self.args.quick_demo else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = runner.run_sparsity_sweep(
            model_name=self.args.model,
            open_domain_texts=open_domain_texts,
            domain_texts=domain_texts,
            test_texts=test_texts,
            sparsity_levels=sparsity_levels,
            domain=self.args.domain
        )
        
        # Analyze results
        logger.info("\nSparsity Analysis Results:")
        logger.info("-" * 50)
        for sparsity, result in results.items():
            if 'error' not in result:
                perplexity = result.get('perplexity', 'N/A')
                logger.info(f"Sparsity {sparsity:.1f}: Perplexity = {perplexity}")
        
        return results
    
    def analyze_model_properties(self):
        """Analyze model size and computational properties"""
        logger.info("=== Analyzing Model Properties ===")
        
        # Load base model for analysis
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch.float16 if self.args.device == "cuda" else torch.float32
        )
        
        # Estimate original model size
        original_size = ModelSizeEstimator.estimate_model_size(model)
        
        # Estimate pruned size
        pruned_size = ModelSizeEstimator.estimate_pruned_size(
            original_size, self.args.sparsity
        )
        
        # Estimate speedup
        speedup_estimates = ModelSizeEstimator.estimate_inference_speedup(
            self.args.sparsity
        )
        
        # Print analysis
        print("\n" + "="*60)
        print("MODEL SIZE ANALYSIS")
        print("="*60)
        print(f"Original Model:")
        print(f"  Total Parameters: {original_size['total_parameters']:,}")
        print(f"  Model Size: {original_size['size_mb']:.1f} MB ({original_size['size_gb']:.2f} GB)")
        print()
        print(f"Pruned Model ({self.args.sparsity:.1%} sparsity):")
        print(f"  Remaining Parameters: {pruned_size['total_parameters']:,}")
        print(f"  Model Size: {pruned_size['size_mb']:.1f} MB ({pruned_size['size_gb']:.2f} GB)")
        print(f"  Compression Ratio: {pruned_size['compression_ratio']:.1f}x")
        print(f"  Space Saved: {pruned_size['space_saved_mb']:.1f} MB")
        print()
        print(f"Estimated Performance Impact:")
        print(f"  Theoretical Speedup: {speedup_estimates['theoretical_speedup']:.1f}x")
        print(f"  Practical Speedup: {speedup_estimates['practical_speedup']:.1f}x")
        print(f"  Memory Reduction: {speedup_estimates['memory_reduction']:.1%}")
        print("="*60)
        
        # Save analysis
        analysis = {
            'original_size': original_size,
            'pruned_size': pruned_size,
            'speedup_estimates': speedup_estimates
        }
        
        analysis_path = self.output_dir / "model_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def generate_final_report(self, all_results: Dict[str, Any]):
        """Generate comprehensive final report"""
        logger.info("=== Generating Final Report ===")
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("D-PRUNER DEMONSTRATION REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Model: {self.args.model}")
        report_lines.append(f"Domain: {self.args.domain}")
        report_lines.append(f"Target Sparsity: {self.args.sparsity:.1%}")
        report_lines.append(f"Device: {self.args.device}")
        report_lines.append(f"Quick Demo Mode: {'Yes' if self.args.quick_demo else 'No'}")
        report_lines.append("")
        
        # Basic results
        if 'basic_results' in all_results:
            basic = all_results['basic_results']
            report_lines.append("BASIC D-PRUNER RESULTS")
            report_lines.append("-" * 50)
            report_lines.append(f"Perplexity: {basic.get('perplexity', 'N/A'):.2f}")
            report_lines.append(f"Pruning Time: {basic.get('pruning_time_seconds', 0):.1f} seconds")
            report_lines.append("")
        
        # Comparison results
        if 'comparison_results' in all_results:
            comparison = all_results['comparison_results']
            report_lines.append("BASELINE COMPARISON")
            report_lines.append("-" * 50)
            
            # Sort by perplexity
            sorted_methods = sorted(
                comparison.items(),
                key=lambda x: x[1].get('perplexity', float('inf'))
            )
            
            for i, (method, metrics) in enumerate(sorted_methods, 1):
                perplexity = metrics.get('perplexity', 0)
                sparsity = metrics.get('sparsity', 0)
                report_lines.append(f"{i}. {method:<15} PPL: {perplexity:.2f}, Sparsity: {sparsity:.1%}")
            report_lines.append("")
        
        # Sparsity analysis
        if 'sparsity_analysis' in all_results:
            sparsity_results = all_results['sparsity_analysis']
            report_lines.append("SPARSITY ANALYSIS")
            report_lines.append("-" * 50)
            for sparsity, result in sorted(sparsity_results.items()):
                if 'error' not in result:
                    perplexity = result.get('perplexity', 'N/A')
                    report_lines.append(f"Sparsity {sparsity:.1f}: Perplexity = {perplexity}")
            report_lines.append("")
        
        # Model analysis
        if 'model_analysis' in all_results:
            analysis = all_results['model_analysis']
            original = analysis['original_size']
            pruned = analysis['pruned_size']
            speedup = analysis['speedup_estimates']
            
            report_lines.append("MODEL ANALYSIS")
            report_lines.append("-" * 50)
            report_lines.append(f"Original Size: {original['size_mb']:.1f} MB")
            report_lines.append(f"Pruned Size: {pruned['size_mb']:.1f} MB")
            report_lines.append(f"Compression: {pruned['compression_ratio']:.1f}x")
            report_lines.append(f"Estimated Speedup: {speedup['practical_speedup']:.1f}x")
            report_lines.append("")
        
        # Performance tracking summary
        pareto_frontier = self.performance_tracker.get_pareto_frontier()
        if pareto_frontier:
            report_lines.append("PARETO FRONTIER (Sparsity vs Performance)")
            report_lines.append("-" * 50)
            for point in pareto_frontier:
                method = point['method']
                sparsity = point['sparsity']
                perplexity = point['perplexity']
                report_lines.append(f"{method}: {sparsity:.1%} sparsity, {perplexity:.2f} PPL")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 50)
        if 'comparison_results' in all_results:
            comparison = all_results['comparison_results']
            best_method = min(comparison.items(), key=lambda x: x[1].get('perplexity', float('inf')))
            report_lines.append(f"Best performing method: {best_method[0]} (PPL: {best_method[1].get('perplexity', 0):.2f})")
        
        report_lines.append(f"D-PRUNER is recommended for domain-specific deployment with {self.args.sparsity:.0%} sparsity.")
        report_lines.append("Consider fine-tuning the pruned model for specific downstream tasks.")
        report_lines.append("")
        
        report_lines.append("=" * 100)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / "final_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Export performance metrics
        metrics_path = self.output_dir / "performance_metrics.csv"
        self.performance_tracker.export_metrics(str(metrics_path))
        
        # Try to generate plots
        try:
            plot_path = self.output_dir / "performance_curves.png"
            self.performance_tracker.plot_performance_curves(str(plot_path))
        except:
            logger.warning("Could not generate performance plots (matplotlib may not be available)")
        
        print("\n" + report_text)
        logger.info(f"Final report saved to {report_path}")
        
        return report_text


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="D-PRUNER Demonstration Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and domain settings
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/DialoGPT-small",
        help="Model name or path for pruning"
    )
    
    parser.add_argument(
        "--domain", 
        type=str, 
        choices=["medical", "healthcare", "legal"], 
        default="medical",
        help="Domain for specialized pruning"
    )
    
    # Pruning parameters
    parser.add_argument(
        "--sparsity", 
        type=float, 
        default=0.5,
        help="Target sparsity ratio (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--lambda-reg", 
        type=float, 
        default=None,
        help="Regularization parameter (auto-selected if not specified)"
    )
    
    # Experimental settings
    parser.add_argument(
        "--compare-baselines", 
        action="store_true",
        help="Run comparison with baseline pruning methods"
    )
    
    parser.add_argument(
        "--sparsity-analysis", 
        action="store_true",
        help="Run analysis across different sparsity levels"
    )
    
    parser.add_argument(
        "--quick-demo", 
        action="store_true",
        help="Run quick demo with reduced dataset sizes"
    )
    
    # Technical parameters
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./dpruner_demo_output",
        help="Output directory for results"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main demonstration function"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle domain aliases
    if args.domain == "healthcare":
        args.domain = "medical"
    
    # Auto-set lambda_reg if not specified
    if args.lambda_reg is None:
        args.lambda_reg = 0.1 if args.domain == "medical" else 0.001
    
    logger.info("Starting D-PRUNER demonstration...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Initialize demo
    demo = DPrunerDemo(args)
    all_results = {}
    
    try:
        # Run basic demonstration
        logger.info("\n" + "="*80)
        basic_results = demo.run_basic_demo()
        all_results['basic_results'] = basic_results
        
        # Run baseline comparison if requested
        if args.compare_baselines:
            logger.info("\n" + "="*80)
            comparison_results = demo.run_comparison_demo()
            all_results['comparison_results'] = comparison_results
        
        # Run sparsity analysis if requested
        if args.sparsity_analysis:
            logger.info("\n" + "="*80)
            sparsity_results = demo.run_sparsity_analysis()
            all_results['sparsity_analysis'] = sparsity_results
        
        # Analyze model properties
        logger.info("\n" + "="*80)
        model_analysis = demo.analyze_model_properties()
        all_results['model_analysis'] = model_analysis
        
        # Generate final report
        logger.info("\n" + "="*80)
        final_report = demo.generate_final_report(all_results)
        
        # Save complete results
        results_path = demo.output_dir / "complete_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Demo completed successfully! Results saved to {demo.output_dir}")
        
        # Print summary
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results directory: {demo.output_dir}")
        print("Key files generated:")
        print(f"  - final_report.txt: Comprehensive analysis report")
        print(f"  - complete_results.json: All numerical results")
        print(f"  - performance_metrics.csv: Performance tracking data")
        print(f"  - pruned_model/: D-PRUNER compressed model")
        
        if args.compare_baselines:
            print(f"  - comparison_report.txt: Baseline comparison report")
            print(f"  - comparison_results.json: Detailed comparison data")
        
        if args.sparsity_analysis:
            print(f"  - sparsity_experiments/: Results across sparsity levels")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


def quick_start_example():
    """Quick start example for users"""
    print("D-PRUNER Quick Start Example")
    print("="*50)
    
    # Simple usage example
    from d_pruner import DPruner, DPrunerConfig, load_sample_data
    
    # Load sample data
    open_domain_texts, domain_texts, test_texts = load_sample_data("medical")
    
    # Configure D-PRUNER
    config = DPrunerConfig(
        model_name="microsoft/DialoGPT-small",  # Small model for demo
        sparsity_ratio=0.3,
        lambda_reg=0.1,
        batch_size=2,
        device="cpu"  # Use CPU for compatibility
    )
    
    # Initialize pruner
    pruner = DPruner(
        config.model_name,
        config.device,
        config.lambda_reg,
        config.alpha
    )
    
    print("Running D-PRUNER pipeline...")
    
    # Step 1: Compute general importance
    print("Step 1: Computing general weight importance...")
    pruner.step1_compute_general_importance(
        open_domain_texts[:20], 
        config.batch_size
    )
    
    # Step 2: Compute dual importance
    print("Step 2: Computing dual importance scores...")
    pruner.step2_compute_dual_importance(
        domain_texts[:20], 
        config.batch_size, 
        1
    )
    
    # Step 3: Prune model
    print("Step 3: Pruning model...")
    pruner.step3_prune_model(config.sparsity_ratio)
    
    # Evaluate
    print("Evaluating pruned model...")
    perplexity = pruner.evaluate_perplexity(test_texts[:10], config.batch_size)
    
    print(f"Final perplexity: {perplexity:.2f}")
    print(f"Achieved sparsity: {config.sparsity_ratio:.1%}")
    
    print("Quick start example completed!")


if __name__ == "__main__":
    # Check if running as quick start
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-start":
        quick_start_example()
    else:
        exit(main())
