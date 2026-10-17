"""Verified table access and fit-only transforms; no comparison runner or gate approval."""

import hashlib
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ecgfounder import CHECKPOINT_BYTES, CHECKPOINT_SHA256
from .ecgfounder_tables import (
    REASONS,
    ROW_DTYPE,
    _freeze_contract,
    _identity,
    _json,
    _sha256,
    _snapshot,
    _validate_result,
    _verify,
)
from .investigation import _compact_prediction
from .modeling import build_pipeline, matrix
from .precursor import LINEAR_SCHEMA, _fit_fixed_logistic, pair_weights

BATCH_ROWS = 4096
MAX_PAIRS = 241


class VerifiedTables:
    """Verify once per comparison, then reuse read-only maps across folds and heads.

    The expected receipt digest and extraction contract must come from the bound
    parent completion manifest/declaration. This verifies data, not fit authorization.
    """

    def __init__(self, arrays, records):
        self.delta, self.nuisance, self.rows = (
            arrays[name] for name in ("delta.npy", "nuisance.npy", "rows.npy")
        )
        self.records = tuple((r["record_id"], r["group_id"], r["patient_id"]) for r in records)
        self._arrays, self._closed = arrays, False

    @classmethod
    def open(
        cls,
        output,
        metadata_rows,
        contract,
        *,
        root,
        receipt_sha256,
        compact_path,
        checkpoint_path,
        feature_schema,
    ):
        output, root = Path(output), Path(root).resolve()
        arrays = {}
        try:
            receipt_path = output / "receipt.json"
            if _sha256(receipt_path) != receipt_sha256:
                raise ValueError("Complete table receipt SHA256 mismatch")
            receipt = json.loads(receipt_path.read_text())
            frozen, records, bindings = _freeze_contract(contract)
            count = frozen["row_count"]
            if (
                receipt["status"] != "complete"
                or receipt["complete"] is not True
                or receipt["rows_written"] != count
                or receipt["row_count"] != count
                or receipt["contract_sha256"] != hashlib.sha256(_json(frozen)).hexdigest()
                or receipt["identity_sha256"] != frozen["identity_sha256"]
                or receipt["records"] != frozen["records"]
                or receipt["reason_names"] != list(REASONS)
                or receipt["expected_files_sha256"] != bindings
            ):
                raise ValueError("Incomplete or different extraction contract")
            _verify(receipt["hashes_before"], bindings)
            _verify(receipt["hashes_after"], bindings)
            for name in (compact_path, checkpoint_path):
                if name not in frozen["files_sha256"]:
                    raise ValueError("Compact/schema and pinned checkpoint must be required bound inputs")
            _verify(_snapshot(bindings, root), bindings)
            if (
                bindings[checkpoint_path] != CHECKPOINT_SHA256
                or (root / checkpoint_path).stat().st_size != CHECKPOINT_BYTES
            ):
                raise ValueError("Pinned checkpoint identity differs")
            compact = json.loads((root / compact_path).read_text())
            if (
                compact["features"] != list(feature_schema)
                or len(set(feature_schema)) != len(feature_schema)
                or not set(LINEAR_SCHEMA) <= set(feature_schema)
                or compact["records"] != frozen["records"]
            ):
                raise ValueError("Compact feature schema or canonical record mapping differs")
            specs = {
                "delta.npy": ((count, 1024), np.dtype("<f4")),
                "nuisance.npy": ((count, 6), np.dtype("<f4")),
                "rows.npy": ((count,), ROW_DTYPE),
            }
            if set(receipt["tables_sha256"]) != set(specs):
                raise ValueError("Exact three fixed tables required")
            for name, (shape, dtype) in specs.items():
                path = output / name
                if (
                    not path.resolve().is_relative_to(output.resolve())
                    or _sha256(path) != receipt["tables_sha256"][name]
                ):
                    raise ValueError("Table path/hash differs")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if not isinstance(values, np.memmap):
                    values.close()
                    raise ValueError("Fixed table must be a read-only NumPy memmap")
                arrays[name] = values
                if (
                    values.shape != shape
                    or values.dtype != dtype
                    or receipt["table_shapes"][name] != list(shape)
                    or _json(receipt["table_dtypes"][name]) != _json(np.lib.format.dtype_to_descr(dtype))
                ):
                    raise ValueError("Fixed table shape/dtype differs")
            metadata, digest = iter(metadata_rows), hashlib.sha256()
            for start in range(0, count, BATCH_ROWS):
                for index in range(start, min(start + BATCH_ROWS, count)):
                    try:
                        expected = _identity(next(metadata), index, records)
                    except StopIteration as error:
                        raise ValueError("Original metadata identity stream incomplete") from error
                    status = arrays["rows.npy"][index]
                    code = int(status["record_index"])
                    if not 0 <= code < len(frozen["records"]):
                        raise ValueError("Unknown table record index")
                    record = frozen["records"][code]
                    accepted = bool(status["accepted"])
                    if (
                        bool(status["evaluated"]) != accepted
                        or not accepted
                        and int(status["valid_chunks"]) != -1
                    ):
                        raise ValueError("Unmeasured original rejection status differs")
                    result = dict(
                        record,
                        row_index=int(status["row_index"]),
                        timestamp=float(status["timestamp"]),
                        accepted=accepted,
                        available=bool(status["available"]),
                        decision_sample=int(status["decision_sample"]),
                        valid_chunks=int(status["valid_chunks"]) if accepted else None,
                        rejections={
                            name: int(value)
                            for name, value in zip(REASONS, status["reason_counts"], strict=True)
                            if value
                        },
                        delta=arrays["delta.npy"][index],
                        nuisance=arrays["nuisance.npy"][index],
                    )
                    _validate_result(result, expected, index)
                    digest.update(_json(expected) + b"\n")
            sentinel = object()
            if next(metadata, sentinel) is not sentinel or digest.hexdigest() != frozen["identity_sha256"]:
                raise ValueError("Original metadata identity stream differs")
            _verify(_snapshot(bindings, root), bindings)
            if _sha256(receipt_path) != receipt_sha256 or any(
                _sha256(output / name) != receipt["tables_sha256"][name] for name in specs
            ):
                raise ValueError("Table/receipt changed during validation")
            return cls(arrays, frozen["records"])
        except BaseException as error:
            for value in arrays.values():
                try:
                    value._mmap.close()
                except BaseException as close_error:
                    error.add_note(f"Read-only table cleanup failed: {close_error}")
            raise

    def batch(self, indices):
        if self._closed:
            raise ValueError("Verified table handle is closed")
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or len(indices) > BATCH_ROWS
            or len(indices)
            and (indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(self.rows)))
        ):
            raise ValueError("Bounded in-range integer row indices required")
        indices = indices.astype(np.int64)
        return self.rows["available"][indices], self.delta[indices], self.nuisance[indices]

    def close(self):
        if not self._closed:
            for value in self._arrays.values():
                value._mmap.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def pair_row_masses(pairs):
    """Accumulate half each original pair's person/event mass on its unique rows."""
    if not 0 < len(pairs) <= MAX_PAIRS:
        raise ValueError("Original fit pair count must be within1..241")
    for pair in pairs:
        if (
            not isinstance(pair.group_id, str)
            or not pair.group_id
            or any(type(i) is not int or i < 0 for i in (pair.positive_index, pair.negative_index))
            or pair.positive_index == pair.negative_index
        ):
            raise ValueError("Invalid original fit pair identity/indices")
    weights, masses = pair_weights(pairs), Counter()
    for pair, weight in zip(pairs, weights, strict=True):
        masses[pair.positive_index] += weight / 2
        masses[pair.negative_index] += weight / 2
    indices = np.asarray(sorted(masses), dtype=np.int64)
    return indices, np.asarray([masses[i] for i in indices]), weights


def _timing_matrix(rows):
    if len(rows) > BATCH_ROWS or len(LINEAR_SCHEMA) != 23:
        raise ValueError("Fixed23 timing schema and bounded rows required")
    if any(any(name not in row for name in LINEAR_SCHEMA) for row in rows):
        raise ValueError("Required timing field missing")
    values = matrix(rows, LINEAR_SCHEMA) if len(rows) else np.empty((0, 46))
    if values.shape != (len(rows), 46) or np.isinf(values).any():
        raise ValueError("Invalid46-column timing matrix")
    return values


@dataclass
class FitTransforms:
    timing_imputer: object
    timing_scaler: StandardScaler
    delta_scaler: StandardScaler
    fit_indices: np.ndarray
    row_masses: np.ndarray

    def transform(self, timing_rows, delta=None):
        values = _timing_matrix(timing_rows)
        timing = (
            self.timing_scaler.transform(self.timing_imputer.transform(values)) if len(values) else values
        )
        if not np.isfinite(timing).all():
            raise ValueError("Nonfinite transformed timing values")
        if delta is None:
            return timing
        delta = np.asarray(delta)
        if delta.shape != (len(values), 1024) or not np.isfinite(delta).all():
            raise ValueError("All1024 delta coordinates must be finite; no imputation/indicators")
        # Match the float64 timing/solver design without rounding standardized deltas to float32.
        scaled = self.delta_scaler.transform(delta.astype(np.float64)) if len(delta) else delta
        candidate = np.hstack((timing, scaled))
        if candidate.shape != (len(values), 1070) or not np.isfinite(candidate).all():
            raise ValueError("Nonfinite1070-column candidate matrix")
        return candidate

    def manifest(self):
        return dict(
            timing_schema=list(LINEAR_SCHEMA),
            timing_columns=46,
            delta_columns=1024,
            delta_order="preclassifier_channel_order",
            candidate_columns=1070,
            fit_indices=self.fit_indices.tolist(),
            row_masses=self.row_masses.tolist(),
            timing_imputer=self.timing_imputer.statistics_.tolist(),
            **{
                name: {key: getattr(value, key).tolist() for key in ("mean_", "var_", "scale_")}
                for name, value in (
                    ("timing_scaler", self.timing_scaler),
                    ("delta_scaler", self.delta_scaler),
                )
            },
        )

    def identity_sha256(self):
        return hashlib.sha256(_json(self.manifest())).hexdigest()


def fit_transforms(timing_rows, data, original_fit_pairs):
    """Fit only already authorized surviving original pair rows; never select pairs."""
    if not isinstance(data, VerifiedTables) or data._closed or len(timing_rows) != len(data.rows):
        raise ValueError("Verified open tables and original row-indexed timing views required")
    indices, masses, _ = pair_row_masses(original_fit_pairs)
    available, delta, _ = data.batch(indices)
    if not available.all():
        raise ValueError("Both original fit-pair members must be common available")
    for pair in original_fit_pairs:
        for index in (pair.positive_index, pair.negative_index):
            if data.records[int(data.rows[index]["record_index"])] != (
                pair.record_id,
                pair.group_id,
                pair.patient_id,
            ):
                raise ValueError("Original fit pair record/group/patient identity differs")
    timing = _timing_matrix([timing_rows[i] for i in indices])
    imputer = build_pipeline("logistic", {"C": 0.1}, 42)["impute"].fit(timing)
    scaler = StandardScaler().fit(imputer.transform(timing), sample_weight=masses)
    delta_scaler = StandardScaler().fit(delta.astype(np.float64), sample_weight=masses)
    fitted = FitTransforms(imputer, scaler, delta_scaler, indices, masses)
    if any(
        not np.isfinite(getattr(value, key)).all()
        for value in (scaler, delta_scaler)
        for key in ("mean_", "var_", "scale_")
    ):
        raise ValueError("Nonfinite fitted preprocessing parameters")
    return fitted


def _fit_rows(timing_rows, data, original_fit_pairs, transforms):
    """Check the supplied original pairs against the already fitted shared transform."""
    if not isinstance(data, VerifiedTables) or data._closed or len(timing_rows) != len(data.rows):
        raise ValueError("Verified open tables and original timing rows required")
    unique, masses, weights = pair_row_masses(original_fit_pairs)
    if not np.array_equal(unique, transforms.fit_indices) or not np.array_equal(
        masses, transforms.row_masses
    ):
        raise ValueError("Transforms must use identical original fit rows and masses")
    indices = np.asarray(
        [p.positive_index for p in original_fit_pairs] + [p.negative_index for p in original_fit_pairs],
        dtype=np.int64,
    )
    available, delta, nuisance = data.batch(indices)
    if not available.all():
        raise ValueError("Original fit members must be common available")
    for pair in original_fit_pairs:
        for index in (pair.positive_index, pair.negative_index):
            if data.records[int(data.rows[index]["record_index"])] != (
                pair.record_id,
                pair.group_id,
                pair.patient_id,
            ):
                raise ValueError("Original fit pair identity differs")
    return [timing_rows[i] for i in indices], delta, nuisance, np.tile(weights, 2)


@dataclass
class PointwiseHead:
    model: object
    transforms: FitTransforms
    kind: str
    scope: str = "primary"
    allowed_groups: frozenset | None = None
    nuisance_imputer: object = None
    nuisance_scaler: object = None

    def _predict(self, timing_rows, available, delta, nuisance):
        logits = np.full(len(available), np.nan)
        if available.any():
            features = [row for row, keep in zip(timing_rows, available, strict=True) if keep]
            design = self.transforms.transform(
                features, delta[available] if self.kind == "candidate" else None
            )
            if self.kind == "nuisance":
                scaled = self.nuisance_scaler.transform(
                    self.nuisance_imputer.transform(nuisance[available].astype(np.float64))
                )
                design = np.hstack((design, scaled))
            predicted = self.model.decision_function(design)
            if not np.isfinite(design).all() or not np.isfinite(predicted).all():
                raise ValueError("Nonfinite pointwise scores")
            logits[available] = predicted
        return logits

    def logits(self, timing_rows, data, indices):
        available, delta, nuisance = data.batch(indices)
        indices = np.asarray(indices, dtype=np.int64)
        if self.allowed_groups is not None and any(
            data.records[int(data.rows[i]["record_index"])][1] not in self.allowed_groups for i in indices
        ):
            raise ValueError("Diagnostic head scoring is restricted to original fit/stop groups")
        return self._predict([timing_rows[i] for i in indices], available, delta, nuisance)


def fit_primary_heads(timing_rows, data, original_fit_pairs, transforms):
    """Fit sequential fixed pointwise heads, never pair differences or a ranking target."""
    features, delta, _, weights = _fit_rows(timing_rows, data, original_fit_pairs, transforms)
    labels = np.repeat([1, 0], len(original_fit_pairs))
    heads = {}
    for name, kind in (("timing_pointwise", "timing"), ("timing_ecgfounder_delta_pointwise", "candidate")):
        design = transforms.transform(features, delta if kind == "candidate" else None)
        model = _fit_fixed_logistic(design, labels, weights, 42)
        heads[name] = PointwiseHead(model, transforms, kind)
    return heads


def orientation_labels(original_fit_pairs, seed):
    """One fair retain/swap label orientation per sorted original fit person."""
    if type(seed) is not int or seed not in range(101, 120) or not original_fit_pairs:
        raise ValueError("Original fit pairs and fixed null seed101..119 required")
    groups = sorted({pair.group_id for pair in original_fit_pairs})
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Original fit person identity required")
    draws = np.random.default_rng(seed).choice([-1.0, 1.0], len(groups))
    orientation = dict(zip(groups, draws, strict=True))
    positive = np.asarray([int(orientation[p.group_id] == 1) for p in original_fit_pairs])
    return np.concatenate((positive, 1 - positive)), orientation


def _control_groups(original_fit_pairs, stop_groups):
    fit = frozenset(pair.group_id for pair in original_fit_pairs)
    stop = frozenset(stop_groups)
    if fit & stop or any(not isinstance(group, str) or not group for group in stop):
        raise ValueError("Disjoint original fit/stop person groups required")
    return fit | stop


def fit_null_heads(timing_rows, data, original_fit_pairs, transforms, *, stop_groups):
    """Yield19 pointwise controls lazily with unchanged real-fit features and weights."""
    allowed = _control_groups(original_fit_pairs, stop_groups)
    features, delta, _, weights = _fit_rows(timing_rows, data, original_fit_pairs, transforms)
    design = transforms.transform(features, delta)
    for seed in range(101, 120):
        labels, orientation = orientation_labels(original_fit_pairs, seed)
        model = _fit_fixed_logistic(design, labels, weights, 42)
        yield seed, orientation, PointwiseHead(model, transforms, "candidate", "fit_stop_only", allowed)


def fit_nuisance_head(timing_rows, data, original_fit_pairs, transforms, *, stop_groups):
    """Fit the fixed timing-plus-six diagnostic with unique-fit-row preprocessing."""
    allowed = _control_groups(original_fit_pairs, stop_groups)
    features, _, nuisance, weights = _fit_rows(timing_rows, data, original_fit_pairs, transforms)
    _, _, unique = data.batch(transforms.fit_indices)
    imputer = build_pipeline("logistic", {"C": 0.1}, 42)["impute"].fit(unique.astype(np.float64))
    scaler = StandardScaler().fit(
        imputer.transform(unique.astype(np.float64)), sample_weight=transforms.row_masses
    )
    design = np.hstack(
        (transforms.transform(features), scaler.transform(imputer.transform(nuisance.astype(np.float64))))
    )
    if design.shape != (2 * len(original_fit_pairs), 52) or not np.isfinite(design).all():
        raise ValueError("Finite52-column nuisance diagnostic required")
    model = _fit_fixed_logistic(design, np.repeat([1, 0], len(original_fit_pairs)), weights, 42)
    return PointwiseHead(model, transforms, "nuisance", "fit_stop_only", allowed, imputer, scaler)


def fit_calibrator(logits, labels):
    """Natural-prevalence scalar-logit version of train._calibrate, without feature copies."""
    logits, labels = np.asarray(logits), np.asarray(labels)
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or not np.isfinite(logits).all()
        or not np.isin(labels, [0, 1]).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError("Aligned finite available logits and both known natural classes required")
    calibrator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("sigmoid", LogisticRegression(C=100.0, max_iter=1000, random_state=42)),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        calibrator.fit(logits.reshape(-1, 1), labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError("Independent sigmoid calibration did not converge")
    calibration_summary(calibrator)
    return calibrator


def calibration_summary(calibrator):
    scaler, sigmoid = calibrator["scale"], calibrator["sigmoid"]
    slope = float(sigmoid.coef_[0, 0] / scaler.scale_[0])
    intercept = float(sigmoid.intercept_[0] - slope * scaler.mean_[0])
    if not np.isfinite([slope, intercept]).all():
        raise ValueError("Nonfinite calibration parameters")
    return dict(
        calibration_c=100.0,
        raw_score_slope=slope,
        raw_score_intercept=intercept,
        positive_slope=slope > 0,
        sample_weighting="none_natural_monitoring_prevalence",
    )


def score_rows(head, calibrator, original_rows, data, indices):
    """Stream every requested original row; common abstentions retain monitoring metadata."""
    if head.scope != "primary" or calibrator is None:
        raise ValueError("Only primary heads have natural calibrated chronology scoring")
    if len(original_rows) != len(data.rows):
        raise ValueError("Complete original row-indexed chronology required")
    iterator = iter(indices)
    while selected := list(islice(iterator, BATCH_ROWS)):
        available, delta, nuisance = data.batch(selected)
        rows = []
        for index in selected:
            row, status = original_rows[index], data.rows[index]
            expected = [
                index,
                *data.records[int(status["record_index"])],
                float(status["timestamp"]),
                bool(status["accepted"]),
            ]
            if _identity(row, index) != expected:
                raise ValueError("Original scoring row identity/acceptance differs from verified table")
            rows.append(row)
        logits = head._predict([row["features"] for row in rows], available, delta, nuisance)
        risks = np.full(len(selected), np.nan)
        if available.any():
            values = calibrator.predict_proba(logits[available, None])[:, 1]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite calibrated risks")
            risks[available] = np.clip(values, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        for row, accepted, logit, risk in zip(rows, available, logits, risks, strict=True):
            yield dict(
                _compact_prediction(row),
                original_accepted=bool(row["accepted"]),
                accepted=bool(accepted),
                raw_logit=float(logit) if accepted else None,
                risk=float(risk) if accepted else None,
            )
