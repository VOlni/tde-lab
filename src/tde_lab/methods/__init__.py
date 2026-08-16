from .standard_fft import StandardFFT, SubSampleFFT
from .median_rdft import MedianRDFT
from .alpha_trimmed import AlphaTrimmedRDFT
from .hodges_lehmann import HodgesLehmannRDFT
from .adaptive_hl import AdaptiveHLRDFT
from .cwmedian import CWMedianRDFT
from .dct_prefilter import DCTPreFilter
from .distance_base import (
    DistanceMethod,
    EuclideanPowerDistance, MinkowskiDistance,
    CanberraDistance, BrayCurtisDistance, HellingerDistance,
    MahalanobisIMMSEDistance, CosineDistance, PearsonDistance,
)
from .base import BaseMethod, MCFResult

ALL_METHODS = {
    # correlation / robust-DFT family (argmax of |MCF|)
    "standard": StandardFFT,
    "subsample": SubSampleFFT,
    "median": MedianRDFT,
    "atrim": AlphaTrimmedRDFT,
    "hl": HodgesLehmannRDFT,
    "adhl": AdaptiveHLRDFT,
    "cwmedian": CWMedianRDFT,
    # distance-metric family (argmin of S_e(j) over circular lags)
    "dist-l1": EuclideanPowerDistance,
    "dist-pow05": EuclideanPowerDistance,
    "dist-pow15": EuclideanPowerDistance,
    "dist-mink1": MinkowskiDistance,
    "dist-mink2": MinkowskiDistance,
    "dist-canberra": CanberraDistance,
    "dist-braycurtis": BrayCurtisDistance,
    "dist-hellinger": HellingerDistance,
    "dist-mahalanobis": MahalanobisIMMSEDistance,
    "dist-cosine": CosineDistance,
    "dist-pearson": PearsonDistance,
}

# 'all' resolves to this — the MATLAB comparison plots also excluded the
# degenerate immse-based Mahalanobis variant
DEFAULT_KEYS = [k for k in ALL_METHODS if k != "dist-mahalanobis"]


def build_method(name: str, method_config=None, with_dct: bool = False) -> BaseMethod:
    """
    Instantiate a method by key string.

    Parameters
    ----------
    name          : one of the ALL_METHODS keys
    method_config : MethodConfig (optional, for trim_percent, window,
                    lag_limit, normal_halfwidth etc.)
    with_dct      : if True, wrap in DCTPreFilter
    """
    from tde_lab.config.settings import MethodConfig
    cfg = method_config or MethodConfig()

    key = name.lower().strip()
    if key not in ALL_METHODS:
        raise ValueError(f"Unknown method {name!r}. Choose from: {list(ALL_METHODS)}")

    dist_kwargs = {"lag_limit": cfg.lag_limit, "normal_halfwidth": cfg.normal_halfwidth}

    if key == "atrim":
        method = AlphaTrimmedRDFT(trim_percent=cfg.trim_percent)
    elif key == "cwmedian":
        method = CWMedianRDFT(window=cfg.cwmedian_window)
    elif key == "dist-l1":
        method = EuclideanPowerDistance(b=1.0, **dist_kwargs)
    elif key == "dist-pow05":
        method = EuclideanPowerDistance(b=0.5, **dist_kwargs)
    elif key == "dist-pow15":
        method = EuclideanPowerDistance(b=1.5, **dist_kwargs)
    elif key == "dist-mink1":
        method = MinkowskiDistance(p=1.0, **dist_kwargs)
    elif key == "dist-mink2":
        method = MinkowskiDistance(p=2.0, **dist_kwargs)
    elif isinstance(ALL_METHODS[key], type) and issubclass(ALL_METHODS[key], DistanceMethod):
        method = ALL_METHODS[key](**dist_kwargs)
    else:
        method = ALL_METHODS[key]()

    if with_dct:
        method = DCTPreFilter(method, beta=cfg.dct_beta)

    return method


__all__ = [
    "StandardFFT", "SubSampleFFT", "MedianRDFT", "AlphaTrimmedRDFT",
    "HodgesLehmannRDFT", "AdaptiveHLRDFT", "CWMedianRDFT", "DCTPreFilter",
    "DistanceMethod",
    "EuclideanPowerDistance", "MinkowskiDistance", "CanberraDistance",
    "BrayCurtisDistance", "HellingerDistance", "MahalanobisIMMSEDistance",
    "CosineDistance", "PearsonDistance",
    "BaseMethod", "MCFResult", "ALL_METHODS", "DEFAULT_KEYS", "build_method",
]
