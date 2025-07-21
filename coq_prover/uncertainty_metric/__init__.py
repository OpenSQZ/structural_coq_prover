"""
Uncertainty Metric Package

This package provides utilities for analyzing uncertainty in Coq theorem proving
through entropy calculations, similarity analysis, and semantic clustering.
"""

from .shared_utils import (
    EntropyCalculator,
    SimilarityCalculator, 
    DataProcessor,
    StatisticsAggregator,
    FileHandler,
    process_entropy_metrics,
    process_tactic_similarities,
    process_embedding_analysis
)

from .logits_entropy import UncertaintyMetricExp
from .semantic_entropy import SemanticEntropy

__all__ = [
    'EntropyCalculator',
    'SimilarityCalculator',
    'DataProcessor',
    'StatisticsAggregator',
    'FileHandler',
    'process_entropy_metrics',
    'process_tactic_similarities',
    'process_embedding_analysis',
    'UncertaintyMetricExp',
    'SemanticEntropy'
]