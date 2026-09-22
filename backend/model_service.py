from pathlib import Path
from typing import Any
import base64
import io
import json
import numpy as np
import pandas as pd

class ModelNotReadyError(RuntimeError):
    pass


class ExplainabilityUnavailableError(RuntimeError):
    """Raised when SHAP cannot explain the loaded pipeline's final estimator."""
    pass

class ModelService:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.pipeline_path = model_dir / "random_forest_pipeline4.joblib"
        self.metadata_path = model_dir / "metadata.json"
        self._pipeline = None
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        default_mapping = {
            "0": "Benign",
            "1": "DoS",
            "2": "DDoS",
            "3": "PortScan",
            "4": "Bot",
            "7": "Brute Force",
            "8": "Web Attack"
        }
        if self.metadata_path.exists():
            try:
                data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if "class_mapping" not in data:
                    data["class_mapping"] = default_mapping
                return data
            except Exception:
                pass
        return {"required_columns": [], "target_column": "label", "positive_label": 1, "class_mapping": default_mapping}

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not self.pipeline_path.exists():
            raise ModelNotReadyError("Model artifact missing. Upload random_forest_pipeline4.joblib to models/.")
        import joblib
        self._pipeline = joblib.load(self.pipeline_path)
        return self._pipeline

    def _get_class_name(self, raw_class: Any) -> str:
        """Convert numeric/raw class indicators to human-readable labels via metadata.json."""
        mapping = self.metadata.get("class_mapping", {})
        str_key = str(raw_class)
        if str_key in mapping:
            return mapping[str_key]
        if isinstance(raw_class, (float, int)) and not isinstance(raw_class, bool):
            int_key = str(int(raw_class))
            if int_key in mapping:
                return mapping[int_key]
        return f"Class {raw_class}" if not str(raw_class).isalpha() else str(raw_class)

    def health(self):
        ready = self.pipeline_path.exists()
        return {"status": "ready" if ready else "model_not_ready", "artifact": self.pipeline_path.name, "required_columns": self.metadata.get("required_columns", [])}

    def validate(self, frame: pd.DataFrame):
        required = self.metadata.get("required_columns", [])
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError("The uploaded CSV contains no rows.")

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.validate(frame)
        pipeline = self._load_pipeline()
        features = frame.drop(columns=[self.metadata.get("target_column", "label")], errors="ignore")
        raw_predictions = pipeline.predict(features)
        output = frame.copy()
        output["prediction"] = [self._get_class_name(p) for p in raw_predictions]
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(features)
            output["confidence"] = probabilities.max(axis=1).round(4)
        return output

    def predict(self, frame: pd.DataFrame):
        output = self.predict_frame(frame)
        counts = output["prediction"].astype(str).value_counts().to_dict()
        return {"status": "ok", "rows": len(output), "class_distribution": counts, "predictions": output.head(100).to_dict(orient="records")}

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def _split_pipeline(self, pipeline):
        """Return (preprocessor, tree_estimator) from a fitted pipeline.

        Supports a plain estimator (no preprocessing steps) as well as an
        sklearn Pipeline where the last step is the tree-based classifier
        and every step before it is preprocessing (scaling, encoding, etc).
        """
        if hasattr(pipeline, "steps"):
            from sklearn.pipeline import Pipeline
            *pre_steps, (last_name, estimator) = pipeline.steps
            preprocessor = Pipeline(pre_steps) if pre_steps else None
            return preprocessor, estimator
        return None, pipeline

    def _find_pca(self, step_obj):
        """Recursively locate fitted PCA or IncrementalPCA step within preprocessor if present."""
        if step_obj is None:
            return None
        if hasattr(step_obj, "components_"):
            return step_obj
        if hasattr(step_obj, "named_steps"):
            for _, sub_step in step_obj.named_steps.items():
                found = self._find_pca(sub_step)
                if found is not None:
                    return found
        if hasattr(step_obj, "steps"):
            for _, sub_step in step_obj.steps:
                found = self._find_pca(sub_step)
                if found is not None:
                    return found
        return None

    def _feature_names(self, preprocessor, fallback_columns):
        """Best-effort feature names after preprocessing, for SHAP plots/tables."""
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                return list(preprocessor.get_feature_names_out())
            except Exception:
                pass
        return list(fallback_columns)

    def explain(self, frame: pd.DataFrame, max_rows: int = 200, top_n: int = 5):
        """Compute SHAP values for uploaded rows against the trained tree model.

        Works on unlabeled/unseen data (no ground-truth column required),
        unlike confusion-matrix/ROC which need true labels. If the pipeline
        uses PCA, SHAP values are mapped backward to the original features using
        the PCA loading matrix.
        """
        self.validate(frame)
        pipeline = self._load_pipeline()
        target_column = self.metadata.get("target_column", "label")
        features = frame.drop(columns=[target_column], errors="ignore")

        if len(features) > max_rows:
            features = features.iloc[:max_rows].reset_index(drop=True)

        preprocessor, estimator = self._split_pipeline(pipeline)

        try:
            import shap
        except ImportError as exc:
            raise ExplainabilityUnavailableError("The 'shap' package is not installed.") from exc

        tree_types = (
            "RandomForestClassifier", "RandomForestRegressor",
            "ExtraTreesClassifier", "GradientBoostingClassifier",
            "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
        )
        if type(estimator).__name__ not in tree_types:
            raise ExplainabilityUnavailableError(
                f"SHAP TreeExplainer does not support estimator type '{type(estimator).__name__}'."
            )

        transformed = preprocessor.transform(features) if preprocessor is not None else features.values
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        explainer = shap.TreeExplainer(estimator)
        raw_shap = explainer.shap_values(transformed)

        raw_classes = getattr(estimator, "classes_", [])
        class_labels = [self._get_class_name(c) for c in raw_classes]

        # Normalize SHAP output across shap versions/estimator types into:
        #   values: ndarray (n_samples, n_features, n_classes)  [n_classes=1 for regression]
        if isinstance(raw_shap, list):
            values = np.stack(raw_shap, axis=-1)
        else:
            arr = np.asarray(raw_shap)
            values = arr if arr.ndim == 3 else arr[:, :, np.newaxis]

        # Check if preprocessor contains PCA/IncrementalPCA
        pca_step = self._find_pca(preprocessor)
        if pca_step is not None and hasattr(pca_step, "components_"):
            # Map PC-space SHAP values backward to original CSV features using PCA loadings matrix W (n_components, n_original_features)
            pca_loadings = pca_step.components_
            if values.ndim == 3 and pca_loadings.shape[1] == features.shape[1]:
                # values: (N, K, C), pca_loadings: (K, P) -> mapped_values: (N, P, C)
                mapped_values = (np.abs(values[:, :, np.newaxis, :]) * np.abs(pca_loadings[np.newaxis, :, :, np.newaxis])).sum(axis=1)
                values = mapped_values
                feature_names = features.columns.tolist()
            else:
                feature_names = self._feature_names(preprocessor, features.columns)
        else:
            feature_names = self._feature_names(preprocessor, features.columns)

        # Pick the class to explain: metadata's positive_label if it matches a
        # known class, otherwise the class with the highest mean |SHAP|.
        positive_label = str(self.metadata.get("positive_label", ""))
        pos_str_mapped = self._get_class_name(positive_label)
        if pos_str_mapped in class_labels:
            class_index = class_labels.index(pos_str_mapped)
        elif str(positive_label) in [str(c) for c in raw_classes]:
            class_index = [str(c) for c in raw_classes].index(str(positive_label))
        else:
            mean_abs_per_class = np.abs(values).mean(axis=(0, 1))
            class_index = int(np.argmax(mean_abs_per_class)) if values.shape[-1] > 1 else 0

        class_values = values[:, :, class_index]

        mean_abs = np.abs(class_values).mean(axis=0)
        importance = sorted(
            [{"feature": name, "mean_abs_shap": round(float(score), 6)}
             for name, score in zip(feature_names, mean_abs)],
            key=lambda item: item["mean_abs_shap"], reverse=True,
        )

        summary_plot_b64 = self._render_summary_plot(values, feature_names, class_labels)

        row_explanations = []
        display_frame = features.reset_index(drop=True)
        for row_idx in range(min(len(class_values), 50)):
            contributions = []
            for col_idx, name in enumerate(feature_names):
                val = class_values[row_idx, col_idx]
                fval = display_frame.iloc[row_idx][name] if name in display_frame.columns else None
                if hasattr(fval, "item"):
                    fval = fval.item()
                contributions.append((name, val, fval))

            sorted_contributions = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:top_n]
            row_explanations.append({
                "row": row_idx,
                "top_features": [
                    {"feature": name, "shap_value": round(float(val), 6), "feature_value": fval}
                    for name, val, fval in sorted_contributions
                ],
            })

        return {
            "status": "ok",
            "rows_explained": len(class_values),
            "explained_class": class_labels[class_index] if class_labels else None,
            "class_labels": class_labels,
            "feature_importance": importance,
            "summary_plot_png_base64": summary_plot_b64,
            "row_explanations": row_explanations,
        }

    def _render_summary_plot(self, values: np.ndarray, feature_names, class_labels) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        fig = plt.figure(figsize=(10, 7))
        try:
            if values.shape[-1] > 1:
                # Multiclass: grouped bar chart of mean |SHAP| per class.
                shap.summary_plot(
                    [values[:, :, i] for i in range(values.shape[-1])],
                    feature_names=feature_names,
                    class_names=class_labels,
                    plot_type="bar",
                    max_display=15,
                    show=False,
                )
            else:
                shap.summary_plot(values[:, :, 0], feature_names=feature_names, max_display=15, show=False)
            plt.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("ascii")
        except Exception:
            plt.close(fig)
            raise

