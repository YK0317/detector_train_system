"""
Command Line Interface for Train System

Provides easy command-line access to train models with configuration files.
"""

import argparse
import sys
from pathlib import Path
import json
import yaml
import logging
import torch

from ..config import UnifiedTrainingConfig, ConfigTemplateManager, ConfigValidator
from ..core.trainer import UnifiedTrainer


def dry_run_training(config_path: str, num_batches: int = 2):
    """Perform a dry run by importing the dedicated dry run module"""
    try:
        from .dry_run import dry_run_training as perform_dry_run
        success, is_external = perform_dry_run(config_path, num_batches)
        return success  # Only return success for backward compatibility
    except ImportError:
        print("❌ Dry run module not available")
        return False


def create_config_template(template_type: str, output_path: str, format: str = None):
    """Create a configuration template file"""
    
    try:
        template = ConfigTemplateManager.get_template(template_type)
    except ValueError as e:
        print(f"❌ {e}")
        print(f"Available templates: blip, generic, torchvision, complete")
        return
    
    output_path = Path(output_path)
    
    # Determine format from file extension or parameter
    if format is None:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            format = 'yaml'
        elif output_path.suffix.lower() == '.json':
            format = 'json'
        else:
            format = 'yaml'
            output_path = output_path.with_suffix('.yaml')
    
    try:
        if format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2, sort_keys=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
        
        print(f"✅ Configuration template created: {output_path}")
        print(f"📋 Template type: {template_type}")
        
        if template_type == 'complete':
            print(f"💡 This template includes all supported fields")
            print(f"   Edit the configuration as needed for your training setup")
        
    except Exception as e:
        print(f"❌ Failed to create template: {e}")

        
        print(f"✅ Configuration template created: {output_path}")
        print(f"📋 Template type: {template_type}")
        
        if template_type == 'complete':
            print(f"📖 This template includes all supported configuration fields.")
            print(f"💡 Use 'ts-complete-config --with-comments' for documented version.")
def validate_config(config_path: str):
    """Validate a configuration file"""
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = UnifiedTrainingConfig.from_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            config = UnifiedTrainingConfig.from_json(config_path)
        else:
            print(f"❌ Unsupported file format: {config_path.suffix}")
            return False
        
        # Validate configuration
        validation_result = ConfigValidator.validate(config)
        
        if validation_result.is_valid:
            print(f"✅ Configuration is valid: {config_path}")
            print(f"   Model: {config.model.name}")
            print(f"   Dataset: {config.data.name}")
            print(f"   Epochs: {config.training.epochs}")
            print(f"   Output: {config.output.experiment_name}")
            
            if validation_result.warnings:
                print("⚠️ Warnings:")
                for warning in validation_result.warnings:
                    print(f"   {warning}")
        else:
            print(f"❌ Configuration validation failed:")
            for error in validation_result.errors:
                print(f"   {error}")
        
        return validation_result.is_valid
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def train_from_config(config_path: str):
    """Train model from configuration file"""
    
    if not validate_config(config_path):
        return
    
    try:
        # Load configuration
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = UnifiedTrainingConfig.from_yaml(config_path)
        else:
            config = UnifiedTrainingConfig.from_json(config_path)
        
        # Create trainer and start training
        trainer = UnifiedTrainer(config)
        results = trainer.train()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best validation accuracy: {results['best_val_accuracy']:.2f}%")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def list_experiments():
    """List all experiments"""
    
    output_dir = Path("training_output")
    
    if not output_dir.exists():
        print("No experiments found.")
        return
    
    experiments = []
    
    for exp_dir in output_dir.iterdir():
        if exp_dir.is_dir():
            exp_info = {"name": exp_dir.name, "path": str(exp_dir)}
            
            # Check for results file
            results_file = exp_dir / "training_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        exp_info["best_val_accuracy"] = results.get("best_val_accuracy", 0)
                        exp_info["timestamp"] = results.get("timestamp", "")
                except:
                    pass
            
            experiments.append(exp_info)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print("📊 Experiments:")
    print("-" * 80)
    for exp in experiments:
        print(f"Name: {exp['name']}")
        if 'best_val_accuracy' in exp:
            print(f"  Best Val Acc: {exp['best_val_accuracy']:.2f}%")
        if 'timestamp' in exp:
            print(f"  Timestamp: {exp['timestamp']}")
        print(f"  Path: {exp['path']}")
        print()


def show_experiment_results(experiment_name: str):
    """Show detailed results for an experiment"""
    
    exp_dir = Path("training_output") / experiment_name
    results_file = exp_dir / "training_results.json"
    
    if not results_file.exists():
        print(f"❌ Experiment not found: {experiment_name}")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Experiment Results: {experiment_name}")
        print("=" * 60)
        print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
        print(f"Total Epochs: {len(results['train_losses'])}")
        print(f"Timestamp: {results['timestamp']}")
        
        if 'config' in results:
            config = results['config']
            print(f"\nModel: {config.get('model', {}).get('name', 'Unknown')}")
            print(f"Dataset: {config.get('data', {}).get('name', 'Unknown')}")
            print(f"Optimizer: {config.get('training', {}).get('optimizer', 'Unknown')}")
            print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 'Unknown')}")
        
        print(f"\nFiles:")
        for file_path in exp_dir.iterdir():
            print(f"  {file_path.name}")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")


def show_system_info():
    """Show system information and available commands"""
    print("🚀 Train System CLI Information")
    print("=" * 50)
    
    # Version and system info
    print(f"📦 Package: train-system v1.0.0")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Available templates
    print(f"\n📋 Available Templates:")
    templates = ["blip", "generic", "torchvision", "complete"]
    for template in templates:
        print(f"   • {template}")
    
    # CLI commands
    print(f"\n⚡ Quick Commands:")
    print(f"   ts-complete-config        Generate complete config template")
    print(f"   ts-template <type>        Generate specific template")
    print(f"   ts-train <config>         Train model from config")
    print(f"   ts-api                    Start API server")
    
    print(f"\n🔧 Full CLI Commands:")
    print(f"   train-system create-template <type> <output>")
    print(f"   train-system validate <config>")
    print(f"   train-system train <config>")
    print(f"   train-system list")
    print(f"   train-system results <experiment>")
    print(f"   train-system api")
    print(f"   train-system info")
    
    print(f"\n💡 Quick Start:")
    print(f"   1. ts-complete-config --with-comments my_config.yaml")
    print(f"   2. Edit my_config.yaml with your settings")
    print(f"   3. ts-train my_config.yaml")
    
    print(f"\n📖 Documentation:")
    print(f"   • Complete config template includes all fields and examples")
    print(f"   • External adapter support for custom model architectures")
    print(f"   • Performance optimizations and monitoring built-in")


def list_components_command(verbose: bool = False):
    """CLI command to list available components"""
    try:
        from ..registry import initialize_registries, list_available_components
        
        print("🔍 Scanning for components...")
        initialize_registries()
        list_available_components(verbose=verbose)
        
    except ImportError:
        print("❌ Registry system not available")
    except Exception as e:
        print(f"❌ Error listing components: {e}")


def scan_components_command(adapter_paths: list, trainer_paths: list, verbose: bool = False):
    """CLI command to scan additional paths"""
    try:
        from ..registry import initialize_registries, scan_additional_paths, list_available_components
        
        print("🔍 Initializing registries...")
        initialize_registries()
        
        if adapter_paths or trainer_paths:
            print(f"🔍 Scanning additional paths...")
            if adapter_paths:
                print(f"   Adapter paths: {adapter_paths}")
            if trainer_paths:
                print(f"   Trainer paths: {trainer_paths}")
                
            scan_additional_paths(adapter_paths, trainer_paths)
        
        print("\n📋 Updated component list:")
        list_available_components(verbose=verbose)
        
    except ImportError:
        print("❌ Registry system not available")
    except Exception as e:
        print(f"❌ Error scanning components: {e}")


def start_api_server():
    """Start the API server"""
    from ..api.server import run_server
    run_server()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Train System CLI - Comprehensive PyTorch Training Framework',
        epilog="""
Examples:
  # Create a complete configuration template with documentation
  train-system create-template complete my_config.yaml
  
  # Quick generate complete config with comments
  ts-complete-config --with-comments my_detailed_config.yaml
  
  # Train a model from configuration
  train-system train my_config.yaml
  
  # Validate configuration without training
  train-system validate my_config.yaml
  
  # List all experiments
  train-system list
  
  # Start API server
  train-system api --port 8080

Available CLI Commands:
  ts-train              - Quick training command
  ts-template           - Template generator  
  ts-complete-config    - Complete config generator
  ts-api               - API server
  train-system         - Full CLI interface
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create template command
    template_parser = subparsers.add_parser('create-template', help='Create configuration template')
    template_parser.add_argument('type', choices=['blip', 'generic', 'torchvision', 'complete'], help='Template type')
    template_parser.add_argument('output', help='Output file path')
    template_parser.add_argument('--format', choices=['yaml', 'json'], default=None, help='Output format (auto-detected from extension if not specified)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config', help='Configuration file path')
    
    # Dry run command
    dry_run_parser = subparsers.add_parser('dry-run', help='Perform a dry run with sample training')
    dry_run_parser.add_argument('config', help='Configuration file path')
    dry_run_parser.add_argument('--batches', type=int, default=2, help='Number of test batches to run (default: 2)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model from configuration')
    train_parser.add_argument('config', help='Configuration file path')
    
    # List experiments command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Show results command
    results_parser = subparsers.add_parser('results', help='Show experiment results')
    results_parser.add_argument('experiment', help='Experiment name')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='localhost', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information and available commands')
    
    # Registry management commands
    list_components_parser = subparsers.add_parser('list-components', help='List all available adapters and trainers')
    list_components_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    scan_parser = subparsers.add_parser('scan', help='Scan additional paths for components')
    scan_parser.add_argument('--adapter-paths', nargs='*', default=[], help='Additional adapter paths to scan')
    scan_parser.add_argument('--trainer-paths', nargs='*', default=[], help='Additional trainer paths to scan')
    scan_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed results')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'create-template':
        create_config_template(args.type, args.output, args.format)
    
    elif args.command == 'validate':
        validate_config(args.config)
    
    elif args.command == 'dry-run':
        dry_run_training(args.config, args.batches)
    
    elif args.command == 'train':
        train_from_config(args.config)
    
    elif args.command == 'list':
        list_experiments()
    
    elif args.command == 'results':
        show_experiment_results(args.experiment)
    
    elif args.command == 'api':
        from ..api.server import TrainingAPI
        api = TrainingAPI(host=args.host, port=args.port)
        api.run(debug=args.debug)
    
    elif args.command == 'info':
        show_system_info()
    
    elif args.command == 'list-components':
        list_components_command(args.verbose)
    
    elif args.command == 'scan':
        scan_components_command(args.adapter_paths, args.trainer_paths, args.verbose)
    
    else:
        parser.print_help()


def train_command():
    """Entry point for ts-train command"""
    parser = argparse.ArgumentParser(description='Train System - Quick Train Command')
    parser.add_argument('config', help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate config, do not train')
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_config(args.config)
    else:
        train_from_config(args.config)


def template_command():
    """Entry point for ts-template command"""
    parser = argparse.ArgumentParser(description='Train System - Template Generator')
    parser.add_argument('type', choices=['blip', 'generic', 'torchvision', 'complete'], help='Template type')
    parser.add_argument('output', help='Output file path')
    parser.add_argument('--format', choices=['yaml', 'json'], default=None, help='Output format (auto-detected from extension if not specified)')
    
    args = parser.parse_args()
    
    create_config_template(args.type, args.output, args.format)


def complete_config_command():
    """Entry point for ts-complete-config command"""
    parser = argparse.ArgumentParser(description='Generate Complete Configuration Template')
    parser.add_argument('output', nargs='?', default='complete_config.yaml', 
                       help='Output file path (default: complete_config.yaml)')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='Output format')
    parser.add_argument('--with-comments', action='store_true', 
                       help='Include comments and documentation (copies from source template)')
    
    args = parser.parse_args()
    
    if args.with_comments:
        # Copy the complete template file with all comments
        from pathlib import Path
        import shutil
        
        source_template = Path(__file__).parent.parent / "configs" / "complete_config_template.yaml"
        output_path = Path(args.output)
        
        if source_template.exists():
            try:
                shutil.copy2(source_template, output_path)
                print(f"✅ Complete configuration template with comments created: {output_path}")
                print(f"📖 This template includes:")
                print(f"   • All supported configuration fields")
                print(f"   • Detailed comments and documentation")
                print(f"   • Usage examples and validation rules")
                print(f"   • External adapter configuration examples")
            except Exception as e:
                print(f"❌ Failed to copy template: {e}")
        else:
            print(f"❌ Source template not found: {source_template}")
            print("Falling back to basic template generation...")
            create_config_template('complete', args.output)
    else:
        # Generate clean template without comments
        create_config_template('complete', args.output)
        print(f"✅ Clean configuration template created: {args.output}")
        print(f"💡 Use --with-comments to include detailed documentation")


if __name__ == "__main__":
    main()
