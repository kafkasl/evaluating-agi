"""LLM Benchmark Matrix — packaged for import."""
import sys, io

# evaluation_harness prints on import; suppress it
_old = sys.stdout; sys.stdout = io.StringIO()
from .evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH,
    MODEL_IDS, BENCH_IDS, MODEL_NAMES, BENCH_NAMES,
    MODEL_IDX, BENCH_IDX, BENCH_CATS,
)
from .all_methods import predict_logit_svd_blend
sys.stdout = _old

# Expose the module refs needed for the N_BENCH monkey-patch in check_novelty
from . import evaluation_harness, all_methods
