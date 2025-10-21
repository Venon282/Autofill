"""Fit metric interfaces and implementations for LES and SAXS samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import traceback
import sys
import numpy as np

FitResult = Optional[Tuple[float, float, float, float, float, float, float, float, float]]


class BaseFitMetric(ABC):
    """Abstract base class for fit metrics."""

    def __init__(self, processes: int) -> None:
        self.processes = max(1, processes)

    @abstractmethod
    def fit_single(self, sample: Tuple) -> FitResult:
        """Fit a single sample and return fit metrics."""

    def safe_fit_single(self, sample: Tuple) -> FitResult:
        """Execute fit while swallowing worker errors."""
        try:
            return self.fit_single(sample)
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            filename, lineno, func, text = tb
            print(f"Warning: Fit failed in {filename}:{lineno} ({func})")
            print(f"  → Code: {text}")
            print(f"  → Exception: {type(e).__name__}: {e}")
            return None


class LesFitMetric(BaseFitMetric):
    """Fit metric for LES signals."""

    def __init__(self, processes: int) -> None:
        super().__init__(processes)

    def fit_single(self, sample: Tuple) -> FitResult:
        """Fit a cylindrical model to a LES sample."""

        from src.validation.utils.fit_cyl_Ag import fit_cyl_Ag

        y_np, wl, true_diam, true_length, true_conc = sample
        length_mm = 1e-3
        results = fit_cyl_Ag(y_np, length_mm, wl)
        if results is None:
            return None
        pred_diam_nm, pred_length_nm, concentration, _ = results

        diam_err = abs(pred_diam_nm - true_diam)
        length_err = abs(pred_length_nm - true_length)
        conc_err = abs(concentration - true_conc)

        return (
            diam_err,
            length_err,
            conc_err,
            pred_diam_nm,
            pred_length_nm,
            concentration,
            true_diam,
            true_length,
            true_conc,
        )


class SaxsFitMetric(BaseFitMetric):
    """Fit metric for SAXS signals."""

    def __init__(
        self,
        processes: int,
        qmin: float,
        qmax: float,
        scale_factor: float,
        use_first_n_points: Optional[int] = None,
    ) -> None:
        super().__init__(processes)
        self.qmin = qmin
        self.qmax = qmax
        self.scale_factor = scale_factor
        self.use_first_n_points = use_first_n_points

    def fit_single(self, sample: Tuple) -> FitResult:
        """Fit a cylindrical model to a SAXS sample."""

        from lmfit import Parameters, minimize
        from sasmodels.core import load_model
        from sasmodels.data import empty_data1D
        from sasmodels.direct_model import DirectModel

        y_np, q_np, true_diam, true_length, true_conc = sample
        assert y_np.ndim == 1 and q_np.ndim == 1, "Input arrays must be 1D."
        assert len(y_np) == len(q_np), "y_np and q_np must have the same length."
        

        if self.use_first_n_points is not None:
            n_points = min(self.use_first_n_points, len(y_np))
            q_fit = q_np[:n_points]
            i_fit = y_np[:n_points]
        else:
            mask = (q_np >= self.qmin) & (q_np <= self.qmax)
            if not np.any(mask):
                return None
            q_fit = q_np[mask]
            i_fit = y_np[mask]

        model = load_model("cylinder")
        data_fit = empty_data1D(q_fit)
        calc_fit = DirectModel(data_fit, model)
        sld_particle = 7.76211e11 * 1e-16
        sld_solvent = 9.39845e10 * 1e-16

        def residual_log(params):
            p = params.valuesdict()
            intensity = calc_fit(
                scale=p["scale"],
                background=p["background"],
                radius=p["radius"],
                length=p["length"],
                sld=sld_particle,
                sld_solvent=sld_solvent,
                radius_pd=0.0,
                length_pd=0.0,
            )
            eps = 1e-30
            return np.log10(np.clip(i_fit, eps, None)) - np.log10(np.clip(intensity, eps, None))

        params = Parameters()
        params.add("scale", value=1, min=1e-10, max=1e15) # changé
        params.add("radius", value=500, min=5, max=1200) # changé
        params.add("length", value=100.0, min=40.0, max=160.0) # changé
        params.add("background", value=0, min = 0.0 , max = 1e-6,vary=True) # changé

        result = minimize(
            residual_log,
            params,
            method="differential_evolution",
            minimizer_kws=dict(
                popsize=400, # changé
                maxiter=300, # changé
                polish=True,
                tol=1e-6,
                updating="deferred",
                workers=1,
            ),
        )
        result = minimize(
            residual_log,
            result.params,
            method="least_squares",
            loss="linear",
            f_scale=0.1,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=80000,
        )

        fitted_scale = result.params["scale"].value
        radius_a = result.params["radius"].value
        length_a = result.params["length"].value

        volume_a3 = np.pi * radius_a**2 * length_a
        volume_cm3 = volume_a3 * 1e-24

        concentration = fitted_scale / volume_cm3 / 1e12
        radius_nm = np.rint(radius_a / 10)
        pred_diam_nm = radius_nm * 2
        pred_length_nm = np.rint(length_a / 10)

        diam_err = abs(pred_diam_nm - true_diam)
        length_err = abs(pred_length_nm - true_length)
        conc_err = abs(concentration - true_conc)

        return (
            diam_err,
            length_err,
            conc_err,
            pred_diam_nm,
            pred_length_nm,
            concentration,
            true_diam,
            true_length,
            true_conc,
        )
