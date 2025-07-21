# Data Extraction Pipeline

This directory contains the data extraction pipeline that processes Coq source code into structured datasets for machine learning. The pipeline transforms raw Coq files into tokenized, augmented, and embedded data suitable for proof generation models.

## Overview

The data extraction system has two main modes:

1. **Data Generation Mode** (`--mode data_generation`): Processes a complete Coq dataset from scratch using topologically sorted package files
2. **New Theorem Mode** (`--mode new_theorem`): Extends an existing base dataset with new external theorems and their premises

## Prerequisites

**Before running data extraction, you must complete the following setup:**

1. **Build Custom Coq Components**: Complete the build process in the `src/` directory to generate the custom OCaml components required for parsing
2. **Configure Coq Paths**: Update `config.json` with the correct paths:
   - `coqc_path`: Path to your Coq compiler binary
   - `coqtop_path`: Path to your Coq toplevel binary
3. **HoTT Library Support** (if needed): 
   - If your dataset includes HoTT (Homotopy Type Theory) libraries, you need to:
     - Install `hoqc` (HoTT Coq compiler)
     - Apply necessary patches to HoTT-related files
     - Set `hoqc_path` in `config.json`
   - Patch files for HoTT support will be provided separately

## Main Entry Point

```bash
# Initialize complete dataset (uses topologically sorted files)
python data_extraction/main.py --mode data_generation

# Add new theorems to existing dataset
python data_extraction/main.py --mode new_theorem
```

### Data Generation Mode Pipeline

1. **Parsing** (`coq_parser.py`): Extract Coq objects from source files
2. **Tokenization** (`coq_tokenization.py`): Assign unique identifiers to definitions
3. **Data Argumentation** (`data_arg_infer.py`): Generate natural language descriptions via LLM
4. **Embedding Generation** (`embedding_infer.py`): Create semantic embeddings for retrieval

**First-time Data Generation**: For initial dataset creation, **patch mode is strongly recommended**. Patch mode automatically handles compilation errors by adding fallback logic to the original scripts. When compilation errors occur, the system generates patch files that can be applied using `./patch.sh <src_dir> <patch_dir>` to fix problematic source files.

### New Theorem Mode Pipeline

Extends existing datasets by:
- Processing new theorem files listed in `theorem2proof_file`
- Extracting only new definitions and premises (not proofs)
- Merging with existing def_table, ps_table, and embedding data
- Updating tokenizer with new identifiers

**Note**: Dependencies for new theorems must be manually added to `all_deps.json`. Future versions may use `coqdep` for automatic dependency extraction.

## Individual Components

### `coq_parser.py`
Parses Coq source files and extracts structured objects using our custom OCaml integration. Can be run independently to process specific files or package subsets.

**Key Features**:
- Handles both complete datasets and subsets (train/test splits)
- Extracts definitions, proofs, inductives, fixpoints, and other Coq constructs
- **Patch Mode**: Automatically handles compilation errors by generating patched versions of problematic files with fallback logic

**Patch Mode Workflow**:
1. Enable patch mode during initial data generation (recommended for first-time use)
2. When compilation errors occur, the system creates patched files with additional fallback logic
3. Apply patches using: `./patch.sh <original_source_dir> <patch_output_dir>`
4. Re-run data generation with the patched files

### `coq_tokenization.py` 
Assigns unique token IDs to Coq definitions for consistent reference across the system.

**Process**:
- Initializes tokenizer with global tokens (internal symbols, local variables)
- Processes definition objects and assigns unique IDs (≥1000 for user definitions)
- Filters duplicate definitions and maintains ID consistency
- Generates fallback statistics for token coverage analysis

**Standalone Usage**:
```bash
python data_extraction/coq_tokenization.py --mode your_data_suffix
```

### `data_arg_infer.py`
Generates natural language descriptions of Coq definitions using LLM APIs for enhanced retrievability.

**Features**:
- Async batch processing with configurable batch sizes
- Integrates with various LLM providers (configured in `llm_method.py`)
- Enhances definitions with human-readable explanations

**Standalone Usage**:
```bash
python data_extraction/data_arg_infer.py --input_file /path/to/def_table.jsonl --batch_size 150
```

### `embedding_infer.py`
Creates semantic embeddings for definition retrieval using specialized embedding models.

**Process**:
- Splits large datasets into shards for parallel processing
- Uses multi-GPU processing for efficiency
- Generates embeddings via Linq-Embed or similar models
- Merges shard results into final embedding dataset

**Standalone Usage**:
```bash
python data_extraction/embedding_infer.py --input_file /path/to/def_table_arged.jsonl --total_shards 8
```

## Data Structures

### `coq_data/` Directory
Defines structured representations of Coq objects:

- **`Def_class.py`**: Base definition objects and type information
- **`Ps_class.py`**: Proof state and tactic trace structures  
- **`Definition.py`**: Standard Coq definitions
- **`Proof.py`**: Proof objects and goal states
- **`Inductive.py`**: Inductive type definitions
- **`Fixpoint.py`**: Recursive function definitions
- **`Instance.py`**: Typeclass instances
- **`Ltac.py`**: Tactic definitions
- **`Parser.py`**: Main parser orchestrating all object types

### `coq_tokenize/` Directory
Tokenization system providing unique identifiers:

- **`tokenizer.py`**: Main tokenizer assigning IDs to definitions
- **`glob.py`**: Global token definitions (symbols, tactics, etc.)

Each Coq object receives a unique identifier enabling consistent reference throughout the system. The tokenizer handles:
- Global symbols and internal tactics (IDs < 1000)
- User definitions (IDs ≥ 1000) 
- Local variables and temporary names
- New theorem identifiers (IDs ≥ 1000000)

## Dataset Structure

The pipeline outputs several coordinated files:
- **Definition Table** (`def_table.jsonl`): Structured Coq definitions
- **Proof State Table** (`ps_table.jsonl`): Proof states and tactic traces
- **Argumented Definitions** (`def_table_arged.jsonl`): LLM-enhanced descriptions  
- **Embeddings** (`def_table_emb.jsonl`): Semantic vector representations
- **Tokenizer** (`tokenizer.json`): ID mapping for all definitions

**Configuration Updates**: When using `data_extraction/main.py`, the output file paths are automatically updated in `config.json`. If running individual components separately, you need to manually update the configuration paths to match your actual output locations.

## Custom Dataset Integration

To use your own Coq dataset instead of our provided topologically sorted files:

1. Modify the dataset file paths in `coq_parser.py`
2. Ensure proper topological ordering of packages to handle dependencies
3. Update `all_deps.json` with package dependency information
4. Update `package_mapping.json` with logical mapping information for your packages - this file is essential for proper package resolution and compilation flags
5. Currently, both the topological sorting files and `package_mapping.json` require manual updates - automatic generation is not yet supported

For detailed information about our data structures and methodology, please refer to our paper.