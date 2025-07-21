# Structural Coq Prover

A machine learning-enhanced automatic theorem prover for the Coq formal proof system. This system combines large language models with retrieval-augmented generation to automatically generate formal proofs.

## Requirements

- **Python 3.11+** (recommended)

**Basic Usage** (proof generation with API models):
```bash
pip install -r requirements.txt
```

**Advanced Usage** (fine-tuning and local model deployment):
```bash
pip install -r requirements_ft.txt
```

## Usage Note

**Recommended Usage**: Set up environment first, then run commands:
```bash
# Initialize environment (run once per shell session)
source init_env.sh

# Then run commands normally
python [command]
```

**Alternative**: Run commands with explicit PYTHONPATH:
```bash
PYTHONPATH=. python [command]
```
## Setup Guide

Follow these steps in order to set up the complete system:

### Step 1: Build Modified Coq Compiler (`src/`)

The system requires a modified Coq compiler that outputs structural information during compilation.

1. **Install Prerequisites**:
   ```bash
   # OCaml and OPAM (OCaml Package Manager)
   sudo apt-get install opam build-essential
   
   # For Docker environments:
   opam init --disable-sandboxing -y
   # For regular systems:
   # opam init -y
   
   eval $(opam env)
   opam switch create coq-8.11 4.10.0
   eval $(opam env)
   
   # Install dependencies
   opam install dune num ocamlfind camlp5
   ```

2. **Build Modified Coq**:
   ```bash
   # Automated installation (recommended)
   ./install_coq.sh $HOME/coq-structured
   ```
   
   This script will:
   - Clone Coq 8.11 source code
   - Apply our modifications
   - Build and install the modified Coq
   - Automatically update `config.json` with correct binary paths
   
   **Manual Installation**: If you prefer manual control, see [`src/README.md`](src/README.md) for detailed step-by-step instructions.

   See [`src/README.md`](src/README.md) for detailed installation instructions.

### Step 2: Set Environment Variables

Configure LLM API credentials (required for both data extraction and proof generation):

```bash
# Reasoning model - for main proof generation and critical reasoning
export API_KEY_REASONING="your-reasoning-api-key"
export BASE_URL_REASONING="your-reasoning-base-url"  
export MODEL_REASONING="your-reasoning-model-name"

# Explanation model - for state explanations and auxiliary tasks
export API_KEY_EXPLANATION="your-explanation-api-key"
export BASE_URL_EXPLANATION="your-explanation-base-url"
export MODEL_EXPLANATION="your-explanation-model-name"
```

These environment variables are used throughout the system for:
- Data augmentation during extraction (`data_extraction/`)
- Proof generation and tactic suggestions (`coq_prover/`)
- State explanations and proof summaries

### Step 3: Extract Structural Data (`data_extraction/`)

Process Coq source code to extract definitions, proof states, and training data.

1. **Configure Data Sources**:
   - Update data/package_mapping.json with Coq packages to process (defaults provided for common libraries)
   - Update data/sorted_order_all.json with processing order (defaults provided for common libraries)
   
   **Note**: Default configurations are provided for most common Coq libraries. If you need to process libraries not included in the defaults, you'll need to add them manually. Automatic dependency resolution via `coqdep` will be added in future versions.
   
   **Set Data Directory**: In `config.json`, set `data_dir` to a directory containing your Coq library folders:
   ```json
   {
     "paths": {
       "data_dir": "/path/to/coq-libraries",
       "output_data": "./data/"
     }
   }
   ```
   
   - `data_dir`: Directory containing Coq library folders (e.g., `stdlib/`, `mathcomp/`, `iris/`) to process
   - `output_data`: Directory for generated data files (defaults to `./data/` if not specified)
   - `emb_model_path`: Path to the embedding model (embed data for semantic retrieval)

2. **Run Data Extraction**:
   ```bash
   # Extract structured data from Coq packages
   python data_extraction/main.py --mode data_generation
   
   # Process new theorems (if needed)
   python data_extraction/main.py --mode new_theorem
   ```

   This generates:
   - `def_table.jsonl`: Extracted definitions and their contexts
   - `ps_table.jsonl`: Proof states and transitions
   - `def_table_emb.jsonl`: Semantic embeddings for retrieval
   - Package dependency information

   See [`data_extraction/README.md`](data_extraction/README.md) for detailed pipeline information.

### Step 4: Configure the System

**Configure System Paths**:
```bash
# Validate and explain configuration options
python config_helper.py validate
python config_helper.py explain
```

Update `config.json` with:
- **Paths**: Coq binaries, data files, model locations
- **Flags**: Feature toggles and processing modes  
- **Params**: Search parameters and worker limits

The config helper will guide you through required vs optional settings.

### Step 5: Generate Proofs (`coq_prover/`)

With the modified Coq, extracted data, and configuration complete, you can now generate proofs.

1. **Basic Proof Generation**:
   ```bash
   # Generate proofs for a specific theorem
   python coq_prover/main.py --mode theorem --file path/to/file.v --theorem theorem_name
   
   # Process an entire package
   python coq_prover/main.py --mode package --package package_name
   
   # Continue from existing tactics
   python coq_prover/main.py --mode generate --file path/to/file.v --theorem theorem_name --tactics "tactic1" "tactic2"
   ```

2. **Distributed Evaluation**:
   ```bash
   # Run on test dataset with multiple workers
   python coq_prover/main.py --mode all --shard 0
   ```

   See [`coq_prover/README.md`](coq_prover/README.md) for advanced usage and configuration options.

## System Architecture

The system consists of five main components:

1. **Modified Coq Compiler (`src/`)**: Custom OCaml modifications to Coq 8.11 that extract structural information during compilation and proof checking.

2. **Python Coq Wrapper (`coqc.py`)**: A Python-based wrapper around the modified Coq compiler that handles:
   - **Dependency Resolution**: Automatically resolves and imports required Coq libraries
   - **Error Handling**: Intelligent error filtering and timeout management
   - **Batch Processing**: Efficient parallel compilation of multiple files
   - **State Management**: Handles proof state extraction and validation
   - **Integration**: Seamless integration with the ML pipeline

   This wrapper makes the modified Coq compiler much easier to use from Python code and handles the complex dependency management that would otherwise require manual intervention.

3. **Data Extraction Pipeline (`data_extraction/`)**: Processes Coq source code to create structured datasets for machine learning, including definitions, proof states, and semantic embeddings.

4. **Proof Generation Engine (`coq_prover/`)**: ML-enhanced proof generator using LLMs with retrieval-augmented generation, beam search, and hierarchical error correction.

5. **Configuration Management**: Centralized configuration system with validation and explanation tools.

## Key Features

- **Retrieval-Augmented Generation**: Semantic search for relevant definitions and proof patterns
- **Hierarchical Error Correction**: Multi-level tactic refinement with different error handling strategies
- **Beam Search**: Parallel exploration of multiple proof branches
- **Public Notes System**: Cross-step knowledge accumulation for better context-aware reasoning
- **Distributed Processing**: Scalable evaluation across multiple workers
- **Fine-tuning Support**: Tools for training custom models on extracted proof data

## Model Integration

The system supports multiple LLM backends:
- **API-based**: Huoshan, Aliyun, DeepSeek commercial APIs
- **Local**: VLLM-based local model serving
- **Fine-tuned**: Custom models trained on extracted Coq data

## Configuration Helper

Use the configuration helper to manage your setup:

```bash
# Validate current configuration
python config_helper.py validate

# View all configuration options with explanations  
python config_helper.py explain
```

## Project Structure

```
structural_coq_prover/
├── src/                    # Modified Coq compiler source files
├── coqc.py                # Python wrapper for Coq compiler (dependency resolution)
├── data_extraction/        # Data processing pipeline
├── coq_prover/            # ML proof generation system
├── data/                  # Configuration and test data
├── config.json           # Main configuration file
├── config_helper.py      # Configuration management tool
└── coq_server.py         # FastAPI server for proof generation
```

For detailed instructions on each component, see the README files in the respective subdirectories.

## Authors
[@Yanzhen Lu](https://github.com/yzlu0917), [@Hanbin Yang](https://github.com/esperandote), [@Xiaodie Wang](https://github.com/TwinkleXD),[@Ge Zhang](), [@Biao Li](https://github.com/liqiongyu), [@Chenxu Fu](https://github.com/fuchenxu2008), [@Chao Li](https://github.com/chaolili)

For detailed implementation and methodology, see our paper: https://arxiv.org/pdf/2507.02541
