# Clarity Score Evaluation Framework

## 📖 Overview

This is an experimental framework for evaluating the clarity of concept definitions in the Coq formal proof system. It generates concept definitions using Large Language Models (LLMs) and evaluates the equivalence between generated definitions and original definitions using probability scores, thereby quantifying the impact of different prompt strategies on definition clarity.

## 🏗️ Architecture Design
```
clarity_score_experiment/
├── clarity_score_pipeline.py          # Main experimental pipeline
├── item_process.py                    # Data preprocessing module
├── llm_service.py                     # LLM service wrapper
├── score.py                           # Scoring calculation module
├── clarity_score_config.yaml          # Configuration file
├── prompt/                            # Prompt related modules
│ ├── prompt_gen.py            
│ ├── prompt_format.py         
│ └── prompt.py                
├── data_analysis/                     # Analysis tools
│ ├── statistics.py                 
│ └── get_examples.py 
└── README.md
```

## 📁 Generated Output Structure (Default)

After running the pipeline and analysis tools, the following directories will be created under clarity_score_experiment/:

```
data/                                  # Scoring results
├── clarity_score_result/
└── examples/
```

**Note**: The `data/` directory and its contents are generated during pipeline execution and do not exist in the initial project structure.

## 🚀 Quick Start

### Requirements

- Python 3.11+ (recommended)
- Required dependencies:
  ```bash
  pip install openai asyncio pyyaml tqdm numpy tabulate
  ```

### Environment Variables Setup

Set the following environment variables before running (same as in the main project [README.md](../../README.md#step-2-set-environment-variables)):

```bash
export API_KEY_REASONING="your-reasoning-api-key"
export BASE_URL_REASONING="your-reasoning-base-url"  
export MODEL_REASONING="your-reasoning-model-name"
```

### Configuration File Setup

Modify the path configurations in `clarity_score_config.yaml`:

```yaml
default:
  complete_path: "path/to/your/input/data.jsonl"    # Input data path
  save_path: "path/to/your/output/directory/"       # Output directory
  # Other configurations...
```
**Note**: The `path/to/your/input/data.jsonl` file should contain complete prompt samples obtained during the reasoning process for generating the next tactic step.

### Running Experiments

```bash
# Use default configuration
python clarity_score_experiment/clarity_score_pipeline.py

# Use custom parameters
python clarity_score_experiment/clarity_score_pipeline.py \
  --start_idx 0 \
  --end_idx 100 \
  --concurrent_limit 50 \
  --cases origin_only internal_only intuition_only
```

## 🔧 Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | Configuration file path | `clarity_score_experiment/clarity_score_config.yaml` |
| `--start_idx` | Start index | Value from config file |
| `--end_idx` | End index | Value from config file |
| `--def_num_per_item` | Number of definitions per item | Value from config file |
| `--concurrent_limit` | Concurrency limit | Value from config file |
| `--save_file_name` | Save file name | Value from config file |
| `--cases` | List of experimental cases | Value from config file |
| `--case_target` | Case target type | `global_def_only` |

## 📊 Result Analysis Tools

### Statistical Analysis

Calculate average clarity scores for different methods:

```bash
python clarity_score_experiment/data_analysis/statistics.py
```

### Example Extraction

Extract examples from result files with flexible filtering:

```bash
python clarity_score_experiment/data_analysis/get_examples.py
```

## 🚨 Important Notes

1. **API Limits**: Adjust `concurrent_limit` according to your API service provider's limits
2. **Data Format**: Ensure input data conforms to the expected structured format
3. **Path Configuration**: Modify path settings in the configuration file before use
4. **Environment Variables**: Ensure all required environment variables are properly set