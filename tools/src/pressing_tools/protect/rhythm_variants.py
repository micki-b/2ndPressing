"""Parameter variants of the rhythm protector for benchmarking (val only)."""
from pressing_tools.protect.rhythm import Protector as _P


class Clu(_P):
    name = "rhythm-cluster"
    def __init__(self): super().__init__(cluster_s=0.010)


class CluLM(_P):
    name = "rhythm-cluster-lm10"
    def __init__(self): super().__init__(cluster_s=0.010, level_match_db=10.0)


class CluLMF2(_P):
    name = "rhythm-cluster-lm10-f2"
    def __init__(self): super().__init__(cluster_s=0.010, level_match_db=10.0, future=2.0)


class CluLMF2T28(_P):
    name = "rhythm-cluster-lm10-f2-t2.8"
    def __init__(self): super().__init__(cluster_s=0.010, level_match_db=10.0, future=2.0, theta=2.8)


class CluLMF2T30(_P):
    name = "rhythm-cluster-lm10-f2-t3.0"
    def __init__(self): super().__init__(cluster_s=0.010, level_match_db=10.0, future=2.0, theta=3.0)
