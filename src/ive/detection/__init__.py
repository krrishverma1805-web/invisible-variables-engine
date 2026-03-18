"""Detection layer package — subgroup discovery, HDBSCAN clustering, SHAP, temporal analysis, scoring."""

from ive.detection.clustering import ClusteringResult, HDBSCANClusterer, HDBSCANClustering
from ive.detection.pattern_scorer import PatternScorer, ScoredPattern
from ive.detection.shap_interactions import SHAPInteractionAnalyzer, SHAPResult
from ive.detection.subgroup_discovery import SubgroupDiscoverer, SubgroupDiscovery, SubgroupPattern
from ive.detection.temporal_analysis import TemporalAnalyzer

__all__ = [
    "SubgroupDiscovery",
    "SubgroupDiscoverer",
    "SubgroupPattern",
    "HDBSCANClustering",
    "HDBSCANClusterer",
    "ClusteringResult",
    "SHAPInteractionAnalyzer",
    "SHAPResult",
    "TemporalAnalyzer",
    "PatternScorer",
    "ScoredPattern",
]
