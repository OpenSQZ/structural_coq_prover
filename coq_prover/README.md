# Coq Prover

Neural theorem proving engine for automatic Coq proof generation using large language models.

## Prerequisites

1. **Environment Variables**: Set LLM API credentials (see main README)
2. **Data Pipeline**: Complete data extraction process in `../data_extraction/`
3. **Configuration**: Fill all paths in `config.json` correctly

## Usage

### Main Entry Point

```bash
python coq_prover/main.py --mode <mode> [options]
```

**Modes:**
- `all`: Distributed evaluation on test dataset (requires `--shard`)
- `package`: Process entire Coq package (requires `--package`)
- `theorem`: Generate proof for specific theorem (requires `--file`, `--theorem`)
- `generate`: Validate tactic sequence (requires `--file`, `--theorem`, `--tactics`)
- `new_theorem`: Process external theorems

### Key Files

- **`main.py`**: Command-line entry point for proof generation with different modes
- **`coq_context/proof_generator.py`**: Main orchestration engine integrating beam search, hierarchical refinement, retrieval, and all ML techniques
- **`coq_context/run_tactic.py`**: Tactic execution engine with coqtop interaction, base file content resolution, and proof state management
- **`coq_context/prompt_gen.py`**: Structural prompt construction using advanced data processing techniques and context integration

## Configuration

For detailed configuration management, use the configuration helper:

```bash
# Validate current configuration
python config_helper.py validate

# View all configuration options with explanations
python config_helper.py explain
```

The helper provides comprehensive validation and explanations for all configuration options including paths, flags, and parameters.

## Technical Innovations

### Semantic Retrieval
- **Concept Unfolding**: Leverages our modified Coq compiler to progressively unfold concept definitions during proof search, providing precise semantic understanding of mathematical structures
- **Proof Pattern Mining**: Reuse of similar historical proof traces
- **Multi-modal Embeddings**: Combines internal similarity with external embedding models

### Precise External Reference Resolution
Automatic identification and resolution of external dependencies:
- **Require Statement Generation**: Automatically generates necessary `Require` statements
- **Module Path Resolution**: Precise location of external definitions and theorems
- **Dependency Tracking**: Maintains complete dependency graphs for complex proofs

### Public Notes System
Dynamic knowledge accumulation across proof steps. The system maintains a growing knowledge base of successful strategies and insights that get updated after each proof layer, enabling better context-aware reasoning in subsequent steps.

### Hierarchical Tactic Refinement
Multi-level error correction system:
- **Fine-grained Debugging**: Detailed analysis of tactic failures
- **Iterative Refinement**: Multiple attempts with progressively refined strategies
- **Error Pattern Learning**: Learning from common failure modes

### Advanced Execution Framework
- **Sub-goal Management**: Handles complex multi-goal proofs with branching logic
- **Timeout Recovery**: Resilient execution with intelligent retry mechanisms
- **Distributed Processing**: Scalable evaluation across multiple workers

## Research Modules

### Fine-tuning (`coq_finetune/`)
```bash
bash coq_prover/run_ft.sh         # Full parameter fine-tuning with DeepSpeed
bash coq_prover/run_lora.sh        # Parameter-efficient LoRA adaptation
```

### VLLM Inference Serving
```bash
bash coq_prover/vllm_instances.sh  # Start VLLM servers for fine-tuned models across multiple GPUs
```
For multi-GPU deployment of fine-tuned models. Load balancing across multiple services/machines requires additional configuration.

### Clarity Score Experiment (`clarity_score_experiment/`)
Evaluation framework for measuring definition understanding quality and prompt strategy effectiveness.

### Uncertainty Metrics (`uncertainty_metric/`)
- **Semantic Entropy**: Measures proof diversity for uncertainty estimation
- **Logits Entropy**: Token-level confidence analysis
- **Definition Understanding**: Comprehension confidence scoring