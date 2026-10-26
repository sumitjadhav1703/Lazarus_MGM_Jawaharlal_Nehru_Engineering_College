"""Frozen train-only preprocessing and independent sigmoid calibration."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CORE_ECG_FIELDS, ENGINEERED_ECG_FIELDS, FEATURE_VERSION

ECG_FIELDS = set(CORE_ECG_FIELDS)
CLINICAL_FIELDS = {
    "age",
    "sex_male",
    "bmi",
    "hypertension",
    "diabetes",
    "heart_failure",
    "prior_stroke",
    "vascular_disease",
    "prior_af_ablation",
    "prior_afl",
    "pacemaker",
    "beta_blocker",
    "antiarrhythmic",
    "anticoagulation",
    "echo_lad",
    "echo_lvef",
}


def feature_schema(rows, variant):
    engineered = variant.endswith("_engineered")
    core_variant = variant.removesuffix("_engineered")
    if core_variant not in {"clinical", "ecg", "combined", "personal"} or (
        engineered and core_variant == "clinical"
    ):
        raise ValueError(f"Unknown feature variant: {variant}")
    allowed = set()
    ecg_fields = ECG_FIELDS | set(ENGINEERED_ECG_FIELDS) if engineered else ECG_FIELDS
    if core_variant in {"clinical", "combined", "personal"}:
        allowed |= {f"clinical_{key}" for key in CLINICAL_FIELDS}
        allowed |= {f"clinical_{key}_available" for key in CLINICAL_FIELDS}
    if core_variant != "clinical":
        allowed |= ecg_fields
    if core_variant == "personal":
        allowed |= {f"baseline_{key}_deviation" for key in ecg_fields}
        if engineered:
            allowed |= {f"baseline_{key}_zscore" for key in ecg_fields}
        allowed |= {"baseline_count", "baseline_warmup"}
        # Older numeric baseline exports use the same explicitly enumerated ECG names.
        allowed |= {f"baseline_{key}" for key in ECG_FIELDS}
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return sorted(allowed & keys)


def matrix(rows, schema):
    values = np.asarray(
        [[np.nan if row.get(key) is None else float(row[key]) for key in schema] for row in rows], dtype=float
    )
    # Explicit indicators for every field also cover missingness first seen after training.
    return np.hstack((values, np.isnan(values).astype(float)))


def build_pipeline(family, params, seed):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if family in {"logistic", "elasticnet"}:
        steps += [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params.get("C", 1),
                    l1_ratio=params.get("l1_ratio", 0),
                    solver="saga" if family == "elasticnet" else "lbfgs",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    elif family == "xgboost":
        from xgboost import XGBClassifier

        steps.append(
            (
                "model",
                XGBClassifier(
                    n_estimators=params.get("n_estimators", 80),
                    max_depth=params.get("max_depth", 2),
                    reg_lambda=params.get("reg_lambda", 3),
                    learning_rate=0.05,
                    subsample=1,
                    colsample_bytree=1,
                    n_jobs=1,
                    random_state=seed,
                    eval_metric="logloss",
                ),
            )
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline(steps)


def raw_score(pipeline, x):
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(x))
    p = np.clip(pipeline.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class Predictor:
    """Loads trusted local joblib only; evidence describes uncalibrated model score."""

    def __init__(self, pipeline, calibrator, schema, config, manifest):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.schema = schema
        self.config = config
        self.manifest = manifest
        self.version = manifest["version"]

    @classmethod
    def load(cls, path: Path, *, research=False):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        config = json.loads((path / "config.json").read_text())
        if config.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Frozen feature extraction version differs; re-audit and retrain the package")
        schema = json.loads((path / "feature_schema.json").read_text())["features"]
        with (path / "model.joblib").open("rb") as stream:
            model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest.get("model_sha256") != model_hash:
            raise ValueError("Frozen model contract checksum mismatch; re-export the package")
        package = joblib.load(path / "model.joblib")
        if package.get("schema") != schema or package.get("config") != config:
            raise ValueError("Frozen model contract differs from schema/config sidecars")
        if package.get("cohort_contract") != manifest.get("cohort_contract"):
            raise ValueError("Frozen cohort contract differs from embedded model contract")
        if manifest.get("deployment_ready") is False and not research:
            raise ValueError("This research-only artifact has not passed runtime deployment checks")
        if "predictor" in package:
            predictor = package["predictor"]
            if predictor.schema != schema:
                raise ValueError("Embedded predictor schema differs from frozen contract")
            predictor.config, predictor.manifest, predictor.version = config, manifest, manifest["version"]
            return predictor
        return cls(package["pipeline"], package["calibrator"], schema, config, manifest)

    def probabilities(self, rows):
        score = raw_score(self.pipeline, matrix(rows, self.schema))
        return self.calibrator.predict_proba(score.reshape(-1, 1))[:, 1]

    def predict(self, features):
        x = matrix([features], self.schema)
        risk = float(self.calibrator.predict_proba(raw_score(self.pipeline, x).reshape(-1, 1))[0, 1])
        transformed = self.pipeline[:-1].transform(x)
        names = self.schema + ["missingindicator_" + key for key in self.schema]
        model = self.pipeline["model"]
        if hasattr(model, "coef_"):
            contributions = transformed[0] * model.coef_[0]
        else:
            import xgboost as xgb

            contributions = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[0, :-1]
        ranked = np.argsort(np.abs(contributions))[::-1][:5]
        evidence = [
            {
                "feature": names[i],
                "value": (
                    int(features.get(self.schema[i - len(self.schema)]) is None)
                    if i >= len(self.schema)
                    else features.get(names[i])
                ),
                "contribution": float(contributions[i]),
            }
            for i in ranked
        ]
        return {
            "risk": risk,
            "evidence": evidence,
            "uncertainty": "Research forecast; contributions explain raw model score, not causation. "
            + (
                "Synthetic demonstration, no clinical validation."
                if self.manifest.get("synthetic")
                else "Population and calibration limits apply; see model card."
            ),
        }


def dependency_versions():
    return {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "xgboost", "joblib", "wfdb"]}
