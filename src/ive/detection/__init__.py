"""Detection layer package — subgroup discovery, HDBSCAN clustering, SHAP, temporal analysis, scoring."""
from ive.detection.subgroup_discovery import SubgroupDiscoverer, SubgroupPattern
from ive.detection.clustering import HDBSCANClusterer, ClusteringResult
from ive.detection.shap_interactions import SHAPInteractionAnalyzer, SHAPResult
from ive.detection.temporal_analysis import TemporalAnalyzer
from ive.detection.pattern_scorer import PatternScorer, ScoredPattern

__all__ = [
    "SubgroupDiscoverer", "SubgroupPattern",
    "HDBSCANClusterer", "ClusteringResult",
    "SHAPInteractionAnalyzer", "SHAPResult",
    "TemporalAnalyzer",
    "PatternScorer", "ScoredPattern",
]
