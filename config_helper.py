#!/usr/bin/env python3
"""
Configuration Helper for Structural Coq Prover

Usage:
    python config_helper.py validate    # Validate current configuration
    python config_helper.py explain     # Explain all configuration options
"""

import json
import argparse
import os
from pathlib import Path


class ConfigHelper:
    def __init__(self, config_path: str = "./config.json"):
        self.config_path = config_path
        self.config = self._load_config() if os.path.exists(config_path) else {}
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def validate(self):
        """Validate the current configuration."""
        print("🔍 Validating configuration...")
        valid = True
        
        # Check required sections
        required_sections = ['paths', 'flags', 'params']
        for section in required_sections:
            if section not in self.config:
                print(f"❌ Missing required section: {section}")
                valid = False
        
        if 'paths' in self.config:
            valid &= self._validate_paths()
        
        if 'flags' in self.config:
            valid &= self._validate_flags()
            
        if 'params' in self.config:
            valid &= self._validate_params()
        
        if valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has issues. See above for details.")
    
    def validate4data_generation(self):
        """Validate the current configuration for data generation."""
        print("🔍 Validating configuration for data generation...")
        
        required_paths = ['coqc_path', 'coqtop_path', 'emb_model_path', 'output_data', 'coqc_error_log', 'dep_file', 'ordered_data_file', 'data_dir', 'package_mapping']
        for path in required_paths:
            if path not in self.config['paths']:
                print(f"❌ Missing required path: {path}")
                return False
            path_val = self.config['paths'][path]
            if not path_val.startswith('path/to/'):
                if not os.path.exists(path_val):
                    print(f"❌ {path} not found: {path_val}")
                    return False
                else:
                    print(f"✅ {path}: {path_val}")
        return True
        
    def _validate_paths(self):
        """Validate path configurations."""
        print("\n📂 Validating paths...")
        paths = self.config['paths']
        flags = self.config.get('flags', {})
        valid = True
        
        # Critical paths that must exist
        critical_paths = {
            'coqc_path': 'Coq compiler binary',
            'coqtop_path': 'Coq toplevel binary',
        }
        
        for path_key, description in critical_paths.items():
            if path_key in paths:
                path_val = paths[path_key]
                if not path_val.startswith('path/to/'):  # Not placeholder
                    if not os.path.exists(path_val):
                        print(f"❌ {description} not found: {path_val}")
                        valid = False
                    else:
                        print(f"✅ {description}: {path_val}")
                else:
                    print(f"⚠️  {description} is placeholder: {path_val}")
            else:
                print(f"❌ Missing required path: {path_key}")
                valid = False
        
        # Check fine-tuned model path if use_ft_model is enabled
        if flags.get('use_ft_model', False):
            if 'ft_model_path' in paths:
                ft_model_path = paths['ft_model_path']
                if not ft_model_path.startswith('path/to/'):
                    if not os.path.exists(ft_model_path):
                        print(f"❌ Fine-tuned model not found: {ft_model_path}")
                        valid = False
                    else:
                        print(f"✅ Fine-tuned model: {ft_model_path}")
                else:
                    print(f"❌ Fine-tuned model path is placeholder: {ft_model_path}")
                    valid = False
            else:
                print("❌ use_ft_model is enabled but ft_model_path is missing")
                valid = False
        
        # Essential data files for proof generation
        essential_files = {
            'emb_model_path': 'Embedding model for semantic retrieval',
            'emb_data_path': 'Embedded definition data',
            'dep_file': 'Package dependency information',
            'ordered_data_file': 'Sorted theorem/package order',
            'package_mapping': 'Package mapping information'
        }
        
        for path_key, description in essential_files.items():
            if path_key in paths:
                path_val = paths[path_key]
                if not path_val.startswith('path/to/'):
                    if not os.path.exists(path_val):
                        print(f"❌ {description} not found: {path_val}")
                        valid = False
                    else:
                        print(f"✅ {description}: {path_val}")
                else:
                    print(f"⚠️  {description} is placeholder: {path_val}")
                    valid = False
            else:
                print(f"❌ Missing essential file: {path_key}")
                valid = False
        
        # Essential directory that must exist
        if 'data_dir' in paths:
            data_dir = paths['data_dir']
            if not data_dir.startswith('path/to/'):
                if not os.path.exists(data_dir):
                    print(f"❌ Data directory not found: {data_dir}")
                    valid = False
                else:
                    print(f"✅ Data directory: {data_dir}")
            else:
                print(f"⚠️  Data is placeholder: {data_dir}")
                valid = False
        else:
            print("❌ Missing essential path: data_dir")
            valid = False
        
        # Directory paths - create if needed
        dir_paths = ['proof_log_dir', 'extra_log_dir', 'system_log_dir', 'temp_dir']
        for path_key in dir_paths:
            if path_key in paths:
                path_val = paths[path_key]
                if not path_val.startswith('path/to/'):
                    Path(path_val).mkdir(parents=True, exist_ok=True)
                    print(f"✅ Directory ready: {path_val}")
        
        return valid
    
    def _validate_flags(self):
        """Validate flag configurations."""
        print("\n🚩 Validating flags...")
        flags = self.config['flags']
        
        # Check for conflicting settings
        if flags.get('use_api') and flags.get('use_ft_model'):
            print("⚠️  Both use_api and use_ft_model are enabled. use_api will take precedence.")
        
        # Validate enum values
        valid_use_origin = ['internal', 'origin', 'mixed']
        if 'use_origin' in flags and flags['use_origin'] not in valid_use_origin:
            print(f"❌ Invalid use_origin: {flags['use_origin']}. Must be: {valid_use_origin}")
            return False
        
        valid_reconsider_modes = ['hierarchical', 'normal', 'disabled']
        if 'reconsider_mode' in flags and flags['reconsider_mode'] not in valid_reconsider_modes:
            print(f"❌ Invalid reconsider_mode: {flags['reconsider_mode']}. Must be: {valid_reconsider_modes}")
            return False
        
        print("✅ Flags configuration looks good")
        return True
    
    def _validate_params(self):
        """Validate parameter configurations.""" 
        print("\n⚙️  Validating parameters...")
        params = self.config['params']
        
        # Check reasonable ranges
        checks = [
            ('beam_width', 1, 10),
            ('max_depth', 5, 100),
            ('theorem_parallel_num', 1, 100),
            ('max_coqc_workers', 1, 500)
        ]
        
        for param, min_val, max_val in checks:
            if param in params:
                val = params[param]
                if not isinstance(val, int) or val < min_val or val > max_val:
                    print(f"⚠️  {param} value {val} is outside recommended range [{min_val}, {max_val}]")
                else:
                    print(f"✅ {param}: {val}")
        
        # Validate model selection
        model_use = params.get('model_use', '')
        print(f"✅ Model selection: {model_use}")
        
        return True
    
    def explain(self):
        """Explain all configuration options."""
        explanations = {
            "paths": {
                "coqc_path": "Path to Coq compiler binary (coqc)",
                "coqtop_path": "Path to Coq toplevel binary (coqtop)",
                "proof_log_dir": "Directory for storing proof generation logs",
                "extra_log_dir": "Directory for additional system logs",
                "system_log_dir": "Directory for system-level logs",
                "def_table_path": "Path to definition table JSONL file with extracted definitions",
                "ps_table_path": "Path to proof state table JSONL file with proof states",
                "tokenizer_path": "Path to tokenizer configuration JSON file",
                "emb_model_path": "Path to embedding model for semantic retrieval",
                "emb_data_path": "Path to embedded definition data JSONL file",
                "state_explanation_model_path": "Path to model for natural language proof state explanations",
                "ft_data_dir": "Directory for fine-tuning training data",
                "temp_dir": "Temporary directory for Coq compilation and processing",
                "output_data": "Directory for storing coqc output data",
                "coqc_error_log": "Directory for storing Coq compiler error logs",
                "dep_file": "JSON file containing package dependency information",
                "ordered_data_file": "JSON file with sorted order of theorems/packages",
                "data_dir": "Prefix path for Coq data files and packages",
                "patch_prefix": "Prefix path for patched Coq data files",
                "package_mapping": "JSON file containing package mapping information",
                "hoqc_path": "Path to HoTT Coq compiler binary (hoqc)",
                "ft_model_path": "Path to fine-tuned language model checkpoint",
                "theorem2proof_file": "Text file containing theorems to be proved"
            },
            "flags": {
                "if_background": "Enable background information(similar proof retrieval) in proof generation",
                "if_use_intuition": "Enable intuition-knowledge enrichment in proof generation",
                "simplify_ps": "Simplify proof states for better readability (only in background mode)",
                "plain_prompt": "Use plain text prompts instead of structured prompts",
                "state_encode": "Enable retrieval use proof state directly instead of semantic information",
                "use_origin": "Controls kernel info granularity (internal/origin/mixed)",
                "if_explanation": "Enable natural language proof state explanations",
                "reconsider_mode": "Error correction strategy: hierarchical (each error with context for individual rethinking), normal (all errors together for hint/strategy), disabled (no error correction)",
                "resume_mode": "Resume from previous interrupted runs",
                "use_api": "Use API-based models instead of local models",
                "use_ft_model": "Use fine-tuned models instead of base models",
                "if_reorganize_concisely": "Reorganize structured proof context concisely",
                "new_theorem_mode": "Enable processing of new/external theorems",
                "ablation_mode": "Enable ablation study mode for research",
                "ablation_proof_mode": "Specific ablation strategy for proof generation",
                "ablation_scope": "Scope of ablation study (def_only/all/etc)",
                "if_strategy": "Enable strategic reasoning in proof generation",
                "if_def": "Enable definition retrieval from knowledge base",
                "if_retrieve": "Enable proof pattern retrieval from history",
                "if_proof_trace": "Enable historical proof context tracking",
                "if_public_notes": "Enable cross-step knowledge accumulation"
            },
            "params": {
                "total_shards": "Number of shards for distributed processing",
                "concept_num": "Number of concepts to retrieve for context",
                "blind_num": "Number of blind search attempts",
                "beam_width": "Number of parallel proof search branches",
                "max_depth": "Maximum proof search depth",
                "max_states_workers": "Maximum number of proof state workers",
                "max_theorems_workers": "Maximum number of theorem processing workers",
                "model_use": "LLM backend to use, just for log",
                "max_attempts": "Maximum number of proof generation attempts",
                "max_retries": "Maximum tactic refinement retry attempts",
                "theorem_parallel_num": "Number of theorems to process concurrently",
                "max_coqc_workers": "Maximum number of Coq compiler worker processes"
            }
        }
        
        print("📖 Configuration Options Explained")
        print("=" * 50)
        
        for section_name, section_items in explanations.items():
            print(f"\n[{section_name.upper()}]")
            print("-" * 30)
            
            for key, desc in section_items.items():
                current_val = "Not set"
                if section_name in self.config and key in self.config[section_name]:
                    current_val = self.config[section_name][key]
                
                print(f"• {key:25s} - {desc}")
                print(f"  Current: {current_val}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Configuration helper for Structural Coq Prover")
    parser.add_argument('command', choices=['validate', 'explain'], 
                       help='validate: Check configuration validity | explain: Show all options')
    parser.add_argument('--config', '-c', default='./config.json',
                       help='Path to config.json file (default: ./config.json)')
    
    args = parser.parse_args()
    
    helper = ConfigHelper(args.config)
    
    if args.command == 'validate':
        helper.validate()
    elif args.command == 'explain':
        helper.explain()


if __name__ == "__main__":
    main()