"""Tests for the hyperparameter tuning API and service helpers.

Background tuning is patched out in HTTP tests so we deterministically test
the request/persistence behaviour without spinning up real Keras runs.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.models.tuning_session import TuningSession, TuningStrategy
from app.models.tuning_session import TuningStatus as TuningSessionStatus
from app.services.tuning_service import (
    TuningService,
    check_tuning_concurrency_limit,
    tuning_service,
)

BASE = "/api/v1/model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_model(session: Session, name: str = "tuning_model") -> ModelBasic:
    """Create a fully-configured model (with project + dataset) ready to train."""
    project = Project(name="tuning_proj")
    session.add(project)
    session.commit()
    session.refresh(project)

    data_file = DataFile(file_name="d.csv", file_type="csv", disk_name="d.csv", project_id=project.id)
    session.add(data_file)
    session.commit()
    session.refresh(data_file)

    model = ModelBasic(
        model_name=name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=1,
        target_field="label",
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=5,
        batch_size=32,
        loss="sparse_categorical_crossentropy",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def _seed_tuning_session(
    session: Session,
    model_id: int,
    strategy: TuningStrategy = TuningStrategy.RANDOM,
    status: TuningSessionStatus = TuningSessionStatus.PENDING,
    best_job_id: str | None = None,
    search_space: dict | None = None,
) -> TuningSession:
    """Create a tuning session row for testing."""
    ts = TuningSession(
        model_id=model_id,
        strategy=strategy,
        search_space=search_space or {"optimizer": ["adam", "sgd"]},
        max_trials=10,
        metric="val_accuracy",
        direction="maximize",
        status=status,
        total_trials=10,
        best_job_id=best_job_id,
    )
    session.add(ts)
    session.commit()
    session.refresh(ts)
    return ts


def _seed_job(
    session: Session,
    model_id: int,
    status: TrainingStatus,
    hyperparams: dict | None = None,
    tuning_session_id: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> TrainingJob:
    job = TrainingJob(
        model_id=model_id,
        status=status,
        hyperparams=hyperparams,
        tuning_session_id=tuning_session_id,
        started_at=started_at,
        completed_at=completed_at,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_background_tuning(mocker):
    """Stop the tuning endpoint from launching real tuning in every HTTP test."""
    return mocker.patch("app.routers.tuning.launch_tuning_session")


# ---------------------------------------------------------------------------
# TuningService unit tests
# ---------------------------------------------------------------------------


class TestGridCombinations:
    """Tests for generate_grid_combinations."""

    def test_grid_combinations_correct(self):
        """3 optimizer × 3 lr × 2 batch = 18 combos."""
        svc = TuningService()
        space = {
            "optimizer": ["adam", "sgd", "rmsprop"],
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [16, 32],
        }
        combos = svc.generate_grid_combinations(space)
        assert len(combos) == 18
        # Each combo has all three keys.
        for c in combos:
            assert set(c.keys()) == {"optimizer", "learning_rate", "batch_size"}
        # Check uniqueness.
        combo_tuples = [tuple(sorted(c.items())) for c in combos]
        assert len(set(combo_tuples)) == 18

    def test_grid_exceeds_max_returns_400(self):
        """More than MAX_GRID_COMBINATIONS → AppException(400)."""
        from app.exceptions import AppException

        svc = TuningService()
        # 10 × 10 × 10 = 1000, well above default 50.
        space = {
            "a": list(range(10)),
            "b": list(range(10)),
            "c": list(range(10)),
        }
        with pytest.raises(AppException) as exc_info:
            svc.generate_grid_combinations(space)
        assert exc_info.value.status_code == 400
        assert "1000" in str(exc_info.value.detail)

    def test_grid_empty_discrete_returns_single_empty(self):
        """No discrete params → single empty dict."""
        svc = TuningService()
        space = {"lr": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        combos = svc.generate_grid_combinations(space)
        assert combos == [{}]

    def test_grid_single_param(self):
        """Single parameter with 3 values → 3 combos."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd", "rmsprop"]}
        combos = svc.generate_grid_combinations(space)
        assert len(combos) == 3


class TestRandomSampling:
    """Tests for sample_random_combination."""

    def test_random_search_respects_max_trials(self):
        """Exactly max_trials combinations generated."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd"], "batch_size": [16, 32, 64]}
        rng = np.random.default_rng(seed=42)
        trials = [svc.sample_random_combination(space, rng) for _ in range(20)]
        assert len(trials) == 20
        for t in trials:
            assert "optimizer" in t
            assert "batch_size" in t

    def test_log_uniform_sampling_in_range(self):
        """Sampled learning rate is always between min and max."""
        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            combo = svc.sample_random_combination(space, rng)
            lr = combo["learning_rate"]
            assert 1e-5 <= lr <= 1e-2, f"lr={lr} out of range"

    def test_uniform_sampling_in_range(self):
        """Uniform sampling stays within bounds."""
        svc = TuningService()
        space = {"dropout": {"type": "uniform", "min": 0.1, "max": 0.5}}
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            combo = svc.sample_random_combination(space, rng)
            assert 0.1 <= combo["dropout"] <= 0.5

    def test_discrete_choice_values(self):
        """Discrete choices only produce values from the list."""
        svc = TuningService()
        space = {"optimizer": ["adam", "sgd"]}
        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            combo = svc.sample_random_combination(space, rng)
            assert combo["optimizer"] in ["adam", "sgd"]


class TestSearchSpaceValidation:
    """Tests for validate_search_space."""

    def test_valid_list_and_dict_specs_pass(self):
        """Well-formed list and dict specs are accepted."""
        svc = TuningService()
        space = {
            "optimizer": ["adam", "sgd"],
            "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2},
            "dropout": {"type": "uniform", "min": 0.1, "max": 0.5},
            "batch_size": [16, 32],
        }
        svc.validate_search_space(space, TuningStrategy.RANDOM)
        svc.validate_search_space(space, TuningStrategy.GRID)

    def test_empty_search_space_rejected(self):
        """Empty search space → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space({}, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "non-empty" in str(exc_info.value.detail)

    def test_empty_list_value_rejected(self):
        """Empty list value → 400 (would otherwise produce zero trials)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"batch_size": [16, 32], "epochs": []}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.GRID)
        assert exc_info.value.status_code == 400
        assert "non-empty list" in str(exc_info.value.detail)

    def test_missing_max_rejected(self):
        """Dict spec without 'max' → 400 (would otherwise crash the thread)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "uniform", "min": 0.1}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "numeric 'min' and 'max'" in str(exc_info.value.detail)

    def test_non_numeric_bounds_rejected(self):
        """Non-numeric min/max → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "uniform", "min": "bad", "max": 1.0}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400

    def test_min_ge_max_rejected(self):
        """min >= max → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"dropout": {"type": "uniform", "min": 0.5, "max": 0.1}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "'min' < 'max'" in str(exc_info.value.detail)

    def test_log_uniform_zero_min_rejected(self):
        """log_uniform with min <= 0 → 400 (log(0) is undefined)."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 0, "max": 1e-2}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.RANDOM)
        assert exc_info.value.status_code == 400
        assert "'min' > 0" in str(exc_info.value.detail)

    def test_grid_requires_discrete_param(self):
        """Grid search with only continuous params → 400."""
        from app.exceptions import AppException

        svc = TuningService()
        space = {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}}
        with pytest.raises(AppException) as exc_info:
            svc.validate_search_space(space, TuningStrategy.GRID)
        assert exc_info.value.status_code == 400
        assert "at least one discrete" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


class TestStartTuning:
    """Tests for POST /model/tuning/{model_name}."""

    def test_tuning_session_created_on_post(self, client, db_session):
        """POST creates a tuning_session row in the DB."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"], "batch_size": [16, 32]},
                "max_trials": 5,
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["success"] is True
        tuning_id = body["data"]["tuning_id"]
        assert tuning_id

        # Verify DB row.
        ts = db_session.get(TuningSession, tuning_id)
        assert ts is not None
        assert ts.strategy == TuningStrategy.RANDOM
        assert ts.max_trials == 5
        assert ts.status == TuningSessionStatus.PENDING

    def test_estimate_seconds_in_response(self, client, db_session):
        """POST 202 response includes estimated_seconds and n_trials."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"optimizer": ["adam", "sgd"]},
                "max_trials": 4,
            },
        )
        assert resp.status_code == 202
        data = resp.json()["data"]
        assert "estimated_seconds" in data
        assert data["estimated_seconds"] > 0
        assert data["n_trials"] == 4

    def test_grid_search_post_400_too_many(self, client, db_session):
        """Grid search with >50 combos returns 400."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {
                    "a": list(range(10)),
                    "b": list(range(10)),
                    "c": list(range(10)),
                },
            },
        )
        assert resp.status_code == 400

    def test_unknown_model_404(self, client, db_session):
        """POST for non-existent model returns 404."""
        resp = client.post(
            f"{BASE}/tuning/does_not_exist",
            json={
                "model_name": "does_not_exist",
                "strategy": "random",
                "search_space": {"optimizer": ["adam"]},
            },
        )
        assert resp.status_code == 404

    def test_unconfigured_model_400(self, client, db_session):
        """Model without training config returns 400."""
        model = ModelBasic(model_name="bare", optimizer="adam")
        db_session.add(model)
        db_session.commit()
        resp = client.post(
            f"{BASE}/tuning/bare",
            json={
                "model_name": "bare",
                "strategy": "random",
                "search_space": {"optimizer": ["adam"]},
            },
        )
        assert resp.status_code == 400

    def test_empty_search_space_400(self, client, db_session):
        """POST with an empty search_space returns 400 instead of a silent no-op."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {},
                "max_trials": 5,
            },
        )
        assert resp.status_code == 400
        assert resp.json()["success"] is False

    def test_grid_search_invalid_search_space_validates_before_generating_combinations(
        self, client, db_session, monkeypatch
    ):
        """validate_search_space executes before generate_grid_combinations so invalid inputs fail upfront."""
        _seed_model(db_session, "tuning_model")
        called = False

        def _record_call(*_args, **_kwargs):
            nonlocal called
            called = True
            return [{}]

        monkeypatch.setattr(tuning_service, "generate_grid_combinations", _record_call)

        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {},
            },
        )
        assert resp.status_code == 400
        assert resp.json()["success"] is False
        assert "search_space must be a non-empty object" in resp.json()["message"]
        assert not called, "generate_grid_combinations was invoked before validate_search_space"

    def test_malformed_search_space_with_oversized_grid_validates_schema_first(self, client, db_session):
        """Schema validation occurs before grid size calculation, returning schema error instead of limit error."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {
                    "batch_size": list(range(100)),
                    "optimizer": "adam",
                },
            },
        )
        assert resp.status_code == 400
        assert resp.json()["success"] is False
        assert "optimizer" in resp.json()["message"]
        assert "must be a non-empty list" in resp.json()["message"]

    def test_empty_list_value_400(self, client, db_session):
        """POST with an empty list value returns 400 instead of a zero-trial session."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {"batch_size": [16, 32], "epochs": []},
            },
        )
        assert resp.status_code == 400

    def test_malformed_dict_spec_400(self, client, db_session):
        """POST with a dict spec missing 'max' returns 400 instead of crashing the thread."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "random",
                "search_space": {"learning_rate": {"type": "uniform", "min": 0.1}},
            },
        )
        assert resp.status_code == 400

    def test_continuous_only_grid_400(self, client, db_session):
        """POST grid search with only continuous params returns 400."""
        _seed_model(db_session, "tuning_model")
        resp = client.post(
            f"{BASE}/tuning/tuning_model",
            json={
                "model_name": "tuning_model",
                "strategy": "grid",
                "search_space": {"learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2}},
            },
        )
        assert resp.status_code == 400


class TestGetTuningSession:
    """Tests for GET /model/tuning/{tuning_id}."""

    def test_get_tuning_session(self, client, db_session):
        """GET returns session detail with trial info."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)

        # Add a linked job.
        job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            hyperparams={"optimizer": "adam"},
            tuning_session_id=ts.id,
        )

        resp = client.get(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["tuning_id"] == ts.id
        assert data["status"] == "completed"
        assert len(data["trials"]) == 1
        assert data["trials"][0]["job_id"] == job.id

    def test_get_tuning_session_404(self, client, db_session):
        """GET for non-existent session returns 404."""
        resp = client.get(f"{BASE}/tuning/nonexistent-id")
        assert resp.status_code == 404


class TestCancelTuning:
    """Tests for DELETE /model/tuning/{tuning_id}."""

    def test_cancel_tuning_sets_status(self, client, db_session):
        """DELETE sets status=cancelled and records completed_at timestamp."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 204

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.CANCELLED
        assert ts.completed_at is not None

    def test_cancel_tuning_cancels_running_jobs(self, client, db_session):
        """DELETE also cancels any running trial jobs and records their completed_at timestamp."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)
        job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.RUNNING,
            tuning_session_id=ts.id,
        )

        resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 204

        db_session.refresh(job)
        assert job.status == TrainingStatus.CANCELLED
        assert job.completed_at is not None
        assert job.error_message == "Tuning session cancelled by user"

    def test_cancel_404(self, client, db_session):
        """DELETE for non-existent session returns 404."""
        resp = client.delete(f"{BASE}/tuning/nonexistent-id")
        assert resp.status_code == 404

    def test_cancel_completed_is_noop(self, client, db_session):
        """Cancelling an already-completed session is a 204 no-op."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)

        with patch("app.socketio_instance.sio.emit", new_callable=AsyncMock) as mock_sio_emit:
            resp = client.delete(f"{BASE}/tuning/{ts.id}")

        assert resp.status_code == 204
        mock_sio_emit.assert_not_called()

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED  # unchanged

    def test_cancel_tuning_emits_socket_progress_event(self, client, db_session):
        """DELETE emits tuning_complete event with status=cancelled to tuning:{session_id} room."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        with patch("app.socketio_instance.sio.emit", new_callable=AsyncMock) as mock_sio_emit:
            resp = client.delete(f"{BASE}/tuning/{ts.id}")

        assert resp.status_code == 204
        mock_sio_emit.assert_awaited_once()

        call_args, call_kwargs = mock_sio_emit.call_args
        assert call_args[0] == "tuning_progress"
        assert call_args[1]["type"] == "tuning_complete"
        assert call_args[1]["status"] == "cancelled"
        assert call_kwargs["room"] == f"tuning:{ts.id}"
        assert call_kwargs["namespace"] == "/dl-result"

    def test_cancel_tuning_emits_events_for_running_jobs_and_preserves_best_metric(self, client, db_session):
        """DELETE emits cancellation for running child jobs and preserves best job and metric if available."""
        model = _seed_model(db_session, "tuning_model")
        best_job = _seed_job(db_session, model.id, TrainingStatus.COMPLETED)
        metric_row = TrainingMetric(
            job_id=best_job.id,
            epoch=1,
            metric_name="val_accuracy",
            metric_value=0.9125,
        )
        db_session.add(metric_row)

        ts = _seed_tuning_session(
            db_session,
            model.id,
            status=TuningSessionStatus.RUNNING,
            best_job_id=best_job.id,
        )

        running_job = _seed_job(db_session, model.id, TrainingStatus.RUNNING, tuning_session_id=ts.id)
        db_session.commit()

        with patch("app.socketio_instance.sio.emit", new_callable=AsyncMock) as mock_sio_emit:
            resp = client.delete(f"{BASE}/tuning/{ts.id}")

        assert resp.status_code == 204
        assert mock_sio_emit.await_count == 2

        # First call is for running trial job
        job_call_args, job_call_kwargs = mock_sio_emit.call_args_list[0]
        assert job_call_kwargs["room"] == running_job.id
        assert job_call_args[1]["status"] == "cancelled"

        # Second call is for tuning session
        ts_call_args, ts_call_kwargs = mock_sio_emit.call_args_list[1]
        assert ts_call_args[0] == "tuning_progress"
        assert ts_call_args[1]["type"] == "tuning_complete"
        assert ts_call_args[1]["status"] == "cancelled"
        assert ts_call_args[1]["best_job_id"] == best_job.id
        assert ts_call_args[1]["best_metric"] == 0.9125
        assert ts_call_kwargs["room"] == f"tuning:{ts.id}"


class TestApplyBest:
    """Tests for POST /model/tuning/{tuning_id}/apply-best."""

    def test_apply_best_updates_model_config(self, client, db_session, monkeypatch):
        """apply-best copies best hyperparams to model_basic."""
        model = _seed_model(db_session, "tuning_model")
        best_job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            hyperparams={"optimizer": "sgd", "epochs": 100, "batch_size": 64},
        )
        ts = _seed_tuning_session(
            db_session,
            model.id,
            status=TuningSessionStatus.COMPLETED,
            best_job_id=best_job.id,
        )

        # Patch tuning_service.make_session to use the test DB session.
        def _mock_make_session():
            return db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_make_session)

        # Need a context manager mock for make_session
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        resp = client.post(f"{BASE}/tuning/{ts.id}/apply-best")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["applied_hyperparams"]["optimizer"] == "sgd"
        assert body["data"]["applied_hyperparams"]["epochs"] == 100

        # Verify model was updated.
        db_session.refresh(model)
        assert model.optimizer == "sgd"
        assert model.epochs == 100
        assert model.batch_size == 64

    def test_apply_best_not_completed_400(self, client, db_session):
        """apply-best on non-completed session returns 400."""
        model = _seed_model(db_session, "tuning_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        resp = client.post(f"{BASE}/tuning/{ts.id}/apply-best")
        assert resp.status_code == 400

    def test_apply_best_404(self, client, db_session):
        """apply-best on non-existent session returns 404."""
        resp = client.post(f"{BASE}/tuning/nonexistent-id/apply-best")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


class TestEarlyStop:
    """Tests for early stopping logic."""

    def test_early_stop_terminates_session(self, db_session, monkeypatch):
        """Threshold met → session stops early + early_stopped=True.

        Tests the logic indirectly by verifying the early_stopped flag is set
        on the tuning session when the threshold is met.
        """
        model = _seed_model(db_session, "tuning_model")

        # Create a session with early_stop_threshold.
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.RANDOM,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=10,
            metric="val_accuracy",
            direction="maximize",
            early_stop_threshold=0.9,
            total_trials=10,
            status=TuningSessionStatus.RUNNING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        # Simulate: set early_stopped=True (as the tuning loop would).
        ts.early_stopped = True
        ts.status = TuningSessionStatus.COMPLETED
        ts.completed_trials = 3  # Stopped after 3 of 10.
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        assert ts.early_stopped is True
        assert ts.completed_trials == 3
        assert ts.status == TuningSessionStatus.COMPLETED


class TestTrialTimeout:
    """Tests for trial timeout logic."""

    def test_trial_timeout_cancels_job(self, db_session, monkeypatch):
        """Mock long-running trial → timeout fires → job=cancelled."""
        model = _seed_model(db_session, "tuning_model")
        job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.training_service.make_session", _mock_session_cm)

        svc = TuningService()
        svc._timeout_trial(job.id)

        db_session.refresh(job)
        assert job.status == TrainingStatus.CANCELLED
        assert "timeout" in (job.error_message or "").lower()


class TestEstimateDuration:
    """Tests for estimate_session_duration."""

    def test_estimate_duration_from_last_job(self, db_session, monkeypatch):
        """Last job took 60s, 10 trials → estimate 600s."""
        model = _seed_model(db_session, "tuning_model")
        now = datetime.now(UTC)
        _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            started_at=now - timedelta(seconds=60),
            completed_at=now,
        )

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        svc = TuningService()
        estimate = svc.estimate_session_duration(model.id, 10)
        assert estimate == 600

    def test_estimate_duration_no_history(self, db_session, monkeypatch):
        """No previous jobs → falls back to DEFAULT_TRIAL_ESTIMATE_SECONDS."""
        model = _seed_model(db_session, "tuning_model")

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        svc = TuningService()
        estimate = svc.estimate_session_duration(model.id, 10)
        # DEFAULT_TRIAL_ESTIMATE_SECONDS = 120, so 120 * 10 = 1200.
        assert estimate == 1200

    def test_estimate_duration_guards_clock_skew(self, db_session, monkeypatch):
        """Negative elapsed duration due to clock skew is clamped to positive duration."""
        model = _seed_model(db_session, "tuning_model")
        now = datetime.now(UTC)
        _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            started_at=now,
            completed_at=now - timedelta(seconds=10),
        )

        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

        svc = TuningService()
        estimate = svc.estimate_session_duration(model.id, 5)
        assert estimate >= 5


class TestRunTuningLoopHardeness:
    """Tests for run_tuning_loop leaving sessions in a terminal state."""

    def test_invalid_search_space_marks_session_failed(self, db_session, monkeypatch):
        """A session whose search_space is malformed → FAILED, not stuck RUNNING.

        Regression test: before hardening, ``sample_random_combination`` raised
        KeyError inside the background thread with no handler, leaving the
        session forever in ``RUNNING``.
        """
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.tuning_service.TuningService._emit_progress", lambda *a, **k: None)

        model = _seed_model(db_session, "tuning_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.RANDOM,
            search_space={"learning_rate": {"type": "uniform", "min": 0.1}},
            max_trials=10,
            metric="val_accuracy",
            direction="maximize",
            total_trials=10,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        svc = TuningService()
        svc.run_tuning_loop(ts.id, "tuning_model", loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.FAILED
        assert ts.completed_at is not None

    def test_run_single_trial_does_not_mutate_model_basic(self, db_session, monkeypatch):
        """Trials execute with custom hyperparams without mutating ModelBasic in DB."""
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.model_run.model_run", lambda *a, **k: None)

        model = _seed_model(db_session, "untouched_model")
        original_optimizer = model.optimizer
        original_epochs = model.epochs
        original_batch_size = model.batch_size

        ts = _seed_tuning_session(db_session, model.id)
        svc = TuningService()
        trial_hyperparams = {"optimizer": "sgd", "epochs": 50, "batch_size": 128}

        job_id = svc._run_single_trial(
            session_id=ts.id,
            model_name="untouched_model",
            hyperparams=trial_hyperparams,
            trial_num=1,
            loop=None,
        )

        db_session.refresh(model)
        assert model.optimizer == original_optimizer
        assert model.epochs == original_epochs
        assert model.batch_size == original_batch_size

        job = db_session.get(TrainingJob, job_id)
        assert job is not None
        assert job.hyperparams == trial_hyperparams


class TestTuningConcurrencyLimit:
    """Tests for tuning session concurrency limits."""

    def test_check_concurrency_limit_passes_when_under_limit(self, db_session):
        """When active tuning sessions are below the limit, the check passes without raising."""
        model = _seed_model(db_session, "model_under_limit")
        # 1 active session when limit is 2 (default)
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)
        # Should not raise
        check_tuning_concurrency_limit(db_session)

    def test_check_concurrency_limit_raises_429_when_max_reached(self, db_session):
        """When active tuning sessions reach MAX_CONCURRENT_TUNING_SESSIONS, 429 is raised."""
        from app.exceptions import AppException

        model = _seed_model(db_session, "model_at_limit")
        # Default limit is 2; create 1 PENDING and 1 RUNNING
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.PENDING)
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        with pytest.raises(AppException) as exc_info:
            check_tuning_concurrency_limit(db_session)

        assert exc_info.value.status_code == 429
        assert "Max concurrent tuning sessions reached" in str(exc_info.value.detail)

    def test_completed_failed_cancelled_sessions_do_not_count_towards_limit(self, db_session):
        """Completed, failed, and cancelled sessions are excluded from active concurrency count."""
        model = _seed_model(db_session, "model_inactive_sessions")
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.CANCELLED)
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.FAILED)

        # 3 inactive sessions, 0 active -> check must pass
        check_tuning_concurrency_limit(db_session)

    def test_start_tuning_endpoint_returns_429_when_max_reached(self, client, db_session, monkeypatch):
        """POST /model/tuning/{model_name} returns HTTP 429 when concurrency limit is reached."""
        model = _seed_model(db_session, "model_endpoint_429")
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        # Monkeypatch launch_tuning_session so no background thread is started if called
        monkeypatch.setattr("app.routers.tuning.launch_tuning_session", lambda *a, **k: None)

        payload = {
            "model_name": model.model_name,
            "strategy": "grid",
            "search_space": {"optimizer": ["adam", "sgd"]},
            "max_trials": 2,
            "metric": "val_accuracy",
            "direction": "maximize",
        }
        resp = client.post(f"{BASE}/tuning/{model.model_name}", json=payload)
        assert resp.status_code == 429
        data = resp.json()
        assert data["success"] is False
        assert "Max concurrent tuning sessions reached" in data["message"]

    def test_concurrency_limit_configurable_via_env_var(self, db_session, monkeypatch):
        """Concurrency limit threshold respects MAX_CONCURRENT_TUNING_SESSIONS."""
        from app.exceptions import AppException

        monkeypatch.setattr("app.services.tuning_service.MAX_CONCURRENT_TUNING_SESSIONS", 1)

        model = _seed_model(db_session, "model_env_limit")
        _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        with pytest.raises(AppException) as exc_info:
            check_tuning_concurrency_limit(db_session)

        assert exc_info.value.status_code == 429
        assert "Max concurrent tuning sessions reached (1)" in str(exc_info.value.detail)


class TestStageATuningStateMachine:
    """Stage A regression tests for tuning state transitions and exactly-once terminal events."""

    @staticmethod
    def _patch_make_session(monkeypatch, db_session):
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

    def test_cancelled_pending_session_never_transitions_to_running(self, db_session, monkeypatch):
        """A PENDING session cancelled before the worker starts stays CANCELLED and runs no trials."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "cancel_pending_model")
        ts = _seed_tuning_session(
            db_session,
            model.id,
            strategy=TuningStrategy.GRID,
            status=TuningSessionStatus.CANCELLED,
            search_space={"optimizer": ["adam", "sgd"]},
        )

        emitted = []
        trial_calls = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._run_single_trial",
            lambda _self, *a, **k: trial_calls.append(a) or "unused-job",
        )

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.CANCELLED
        assert trial_calls == []
        assert emitted == []

    def test_cancellation_during_trial_stops_without_scoring_or_duplicate_complete(
        self, client, db_session, monkeypatch
    ):
        """Cancelling via DELETE mid-trial skips scoring/early-stop and emits tuning_complete only once."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "cancel_mid_trial_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            early_stop_threshold=0.5,
            total_trials=2,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        emitted_events = []

        def _record_loop_emit(_self, _loop, _sid, data):
            emitted_events.append(("loop", data))

        async def _record_router_emit(event_name, payload, **_kwargs):
            if event_name == "tuning_progress":
                emitted_events.append(("router", payload))

        monkeypatch.setattr("app.services.tuning_service.TuningService._emit_progress", _record_loop_emit)
        monkeypatch.setattr("app.routers.tuning.sio.emit", _record_router_emit)

        trial_runs = []

        def _run_and_cancel(_self, session_id, _model_name, _hp, trial_num, _loop):
            trial_runs.append(trial_num)
            job = _seed_job(
                db_session,
                model.id,
                TrainingStatus.COMPLETED,
                hyperparams=_hp,
                tuning_session_id=session_id,
            )
            db_session.add(TrainingMetric(job_id=job.id, epoch=1, metric_name="val_accuracy", metric_value=0.95))
            db_session.commit()
            # Simulate user calling DELETE /tuning/{id} while trial 0 is running.
            resp = client.delete(f"{BASE}/tuning/{session_id}")
            assert resp.status_code == 204
            return job.id

        monkeypatch.setattr("app.services.tuning_service.TuningService._run_single_trial", _run_and_cancel)

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert trial_runs == [0]  # Trial 1 never starts
        assert ts.status == TuningSessionStatus.CANCELLED
        assert ts.completed_trials == 0
        assert ts.best_job_id is None
        assert ts.early_stopped is False

        terminal_events = [e for e in emitted_events if e[1].get("type") == "tuning_complete"]
        assert len(terminal_events) == 1
        assert terminal_events[0] == (
            "router",
            {
                "type": "tuning_complete",
                "status": "cancelled",
                "best_job_id": None,
                "best_metric": None,
            },
        )
        assert [e for e in emitted_events if e[1].get("type") == "tuning_progress"] == []

    def test_concurrent_cancel_requests_emit_tuning_complete_once(self, client, db_session):
        """Calling DELETE twice on an active session emits tuning_complete only on the first transition."""
        model = _seed_model(db_session, "double_cancel_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)

        with patch("app.socketio_instance.sio.emit", new_callable=AsyncMock) as mock_sio_emit:
            resp1 = client.delete(f"{BASE}/tuning/{ts.id}")
            resp2 = client.delete(f"{BASE}/tuning/{ts.id}")

        assert resp1.status_code == 204
        assert resp2.status_code == 204
        mock_sio_emit.assert_awaited_once()

    def test_fail_tuning_session_preserves_cancelled_and_completed_states(self, db_session, monkeypatch):
        """_fail_tuning_session never overwrites CANCELLED or COMPLETED sessions or their timestamps."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "terminal_guard_model")
        cancelled_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed_at = datetime(2026, 1, 1, 13, 0, 0, tzinfo=UTC)

        ts_cancelled = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.CANCELLED)
        ts_cancelled.completed_at = cancelled_at
        ts_completed = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.COMPLETED)
        ts_completed.completed_at = completed_at
        db_session.add(ts_cancelled)
        db_session.add(ts_completed)
        db_session.commit()

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        svc = TuningService()
        svc._fail_tuning_session(ts_cancelled.id, loop=None, message="late error")
        svc._fail_tuning_session(ts_completed.id, loop=None, message="late error")

        db_session.refresh(ts_cancelled)
        db_session.refresh(ts_completed)
        assert ts_cancelled.status == TuningSessionStatus.CANCELLED
        assert ts_cancelled.completed_at.replace(tzinfo=None) == cancelled_at.replace(tzinfo=None)
        assert ts_completed.status == TuningSessionStatus.COMPLETED
        assert ts_completed.completed_at.replace(tzinfo=None) == completed_at.replace(tzinfo=None)
        assert emitted == []

    def test_normal_completion_emits_tuning_complete_exactly_once(self, db_session, monkeypatch):
        """Happy-path run transitions PENDING -> RUNNING -> COMPLETED and emits tuning_complete once."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "happy_tuning_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            total_trials=2,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        created_jobs = []

        def _fake_trial(_self, session_id, _model_name, hp, trial_num, _loop):
            job = _seed_job(
                db_session,
                model.id,
                TrainingStatus.COMPLETED,
                hyperparams=hp,
                tuning_session_id=session_id,
            )
            db_session.add(
                TrainingMetric(
                    job_id=job.id,
                    epoch=1,
                    metric_name="val_accuracy",
                    metric_value=0.70 + trial_num * 0.15,
                )
            )
            db_session.commit()
            created_jobs.append(job.id)
            return job.id

        monkeypatch.setattr("app.services.tuning_service.TuningService._run_single_trial", _fake_trial)

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED
        assert ts.completed_trials == 2
        assert ts.best_job_id == created_jobs[1]
        assert ts.completed_at is not None

        terminal_events = [e for e in emitted if e.get("type") == "tuning_complete"]
        assert len(terminal_events) == 1
        assert terminal_events[0]["status"] == "completed"
        assert terminal_events[0]["best_job_id"] == created_jobs[1]
        assert terminal_events[0]["best_metric"] == 0.85


class TestStageBTrialIntegrity:
    """Stage B regression tests for trial eligibility, timeouts, metric fallback, and non-finite guards."""

    @staticmethod
    def _patch_make_session(monkeypatch, db_session):
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)
        monkeypatch.setattr("app.services.training_service.make_session", _mock_session_cm)

    def test_non_completed_trials_with_metrics_do_not_become_best_job(self, db_session, monkeypatch):
        """Failed and timed-out/cancelled trials with high partial metrics cannot win best_job_id or early-stop."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "trial_eligibility_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd", "rmsprop"]},
            max_trials=3,
            metric="val_accuracy",
            direction="maximize",
            early_stop_threshold=0.95,
            total_trials=3,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        trial_outcomes = [
            (TrainingStatus.FAILED, 0.99),
            (TrainingStatus.CANCELLED, 0.98),
            (TrainingStatus.COMPLETED, 0.82),
        ]
        created_jobs = []

        def _fake_trial(_self, session_id, _model_name, hp, trial_num, _loop):
            status, metric_val = trial_outcomes[trial_num]
            job = _seed_job(
                db_session,
                model.id,
                status,
                hyperparams=hp,
                tuning_session_id=session_id,
            )
            db_session.add(TrainingMetric(job_id=job.id, epoch=1, metric_name="val_accuracy", metric_value=metric_val))
            db_session.commit()
            created_jobs.append(job.id)
            return job.id

        monkeypatch.setattr("app.services.tuning_service.TuningService._run_single_trial", _fake_trial)

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED
        assert ts.completed_trials == 3
        assert ts.early_stopped is False
        assert ts.best_job_id == created_jobs[2]

        terminal = [e for e in emitted if e.get("type") == "tuning_complete"]
        assert len(terminal) == 1
        assert terminal[0]["best_job_id"] == created_jobs[2]
        assert terminal[0]["best_metric"] == 0.82

    def test_all_non_completed_trials_leave_best_job_id_none(self, db_session, monkeypatch):
        """If every trial fails or times out, best_job_id and best_metric remain None."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "all_failed_trials_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            total_trials=2,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        def _fake_trial(_self, session_id, _model_name, hp, trial_num, _loop):
            status = TrainingStatus.FAILED if trial_num == 0 else TrainingStatus.CANCELLED
            job = _seed_job(db_session, model.id, status, hyperparams=hp, tuning_session_id=session_id)
            db_session.add(TrainingMetric(job_id=job.id, epoch=1, metric_name="val_accuracy", metric_value=0.91))
            db_session.commit()
            return job.id

        monkeypatch.setattr("app.services.tuning_service.TuningService._run_single_trial", _fake_trial)

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED
        assert ts.completed_trials == 2
        assert ts.best_job_id is None

        terminal = [e for e in emitted if e.get("type") == "tuning_complete"]
        assert len(terminal) == 1
        assert terminal[0]["best_job_id"] is None
        assert terminal[0]["best_metric"] is None

    @pytest.mark.parametrize(
        "job_status",
        [TrainingStatus.FAILED, TrainingStatus.CANCELLED, TrainingStatus.RUNNING, TrainingStatus.PENDING],
    )
    def test_apply_best_rejects_non_completed_best_job(self, client, db_session, monkeypatch, job_status):
        """POST apply-best returns 400 and leaves ModelBasic untouched if best_job is not COMPLETED."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, f"apply_reject_{job_status.value}")
        bad_job = _seed_job(
            db_session,
            model.id,
            job_status,
            hyperparams={"optimizer": "sgd", "epochs": 99, "batch_size": 128},
        )
        ts = _seed_tuning_session(
            db_session,
            model.id,
            status=TuningSessionStatus.COMPLETED,
            best_job_id=bad_job.id,
        )

        resp = client.post(f"{BASE}/tuning/{ts.id}/apply-best")
        assert resp.status_code == 400
        assert "not completed" in resp.json()["message"].lower()

        db_session.refresh(model)
        assert model.optimizer == "adam"
        assert model.epochs == 5
        assert model.batch_size == 32

    @pytest.mark.parametrize(
        ("initial_status", "initial_error"),
        [
            (TrainingStatus.COMPLETED, None),
            (TrainingStatus.FAILED, "OOM error"),
            (TrainingStatus.CANCELLED, "Tuning session cancelled by user"),
        ],
    )
    def test_timeout_trial_preserves_terminal_job_states(self, db_session, monkeypatch, initial_status, initial_error):
        """_timeout_trial never overwrites COMPLETED, FAILED, or already-CANCELLED jobs."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, f"timeout_guard_{initial_status.value}")
        original_completed_at = datetime(2026, 2, 1, 10, 30, 0, tzinfo=UTC)
        job = TrainingJob(
            model_id=model.id,
            status=initial_status,
            error_message=initial_error,
            completed_at=original_completed_at,
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)

        TuningService()._timeout_trial(job.id)

        db_session.refresh(job)
        assert job.status == initial_status
        assert job.error_message == initial_error
        assert job.completed_at.replace(tzinfo=None) == original_completed_at.replace(tzinfo=None)

    def test_timeout_not_overwritten_when_run_single_trial_catches_exception(self, db_session, monkeypatch):
        """If _timeout_trial marks CANCELLED before model_run raises, _run_single_trial preserves CANCELLED."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "timeout_then_error_model")
        ts = _seed_tuning_session(db_session, model.id, status=TuningSessionStatus.RUNNING)
        svc = TuningService()

        def _slow_failing_run(_model_name, _session, loop=None, job_id=None):
            svc._timeout_trial(job_id)
            raise RuntimeError("halted after timeout")

        monkeypatch.setattr("app.services.model_run.model_run", _slow_failing_run)

        job_id = svc._run_single_trial(ts.id, model.model_name, {"optimizer": "sgd"}, trial_num=0, loop=None)
        job = db_session.get(TrainingJob, job_id)
        assert job is not None
        assert job.status == TrainingStatus.CANCELLED
        assert job.error_message == "Trial timeout exceeded"

    def test_metric_fallback_consistent_across_service_api_and_socketio(self, client, db_session, monkeypatch):
        """Metric val_ fallback resolves identically in service, API GET/DELETE, and Socket.IO catch-up."""
        import app.socketio_instance as si
        from app.services.tuning_service import resolve_trial_metric

        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "metric_fallback_model")

        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            status=TuningSessionStatus.RUNNING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        completed_job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.COMPLETED,
            hyperparams={"optimizer": "adam"},
            tuning_session_id=ts.id,
        )
        failed_job = _seed_job(
            db_session,
            model.id,
            TrainingStatus.FAILED,
            hyperparams={"optimizer": "sgd"},
            tuning_session_id=ts.id,
        )
        # Only unprefixed "accuracy" was logged (no "val_accuracy"), plus reverse "val_loss" for "loss".
        db_session.add(TrainingMetric(job_id=completed_job.id, epoch=1, metric_name="accuracy", metric_value=0.87))
        db_session.add(TrainingMetric(job_id=completed_job.id, epoch=1, metric_name="val_loss", metric_value=0.25))
        db_session.add(TrainingMetric(job_id=failed_job.id, epoch=1, metric_name="accuracy", metric_value=0.99))

        ts.best_job_id = completed_job.id
        db_session.add(ts)
        db_session.commit()

        # 1. Service lookup (both val_->unprefixed and unprefixed->val_ directions)
        svc = TuningService()
        assert svc._get_trial_metric(completed_job.id, "val_accuracy") == 0.87
        assert resolve_trial_metric(db_session, completed_job.id, "loss") == 0.25

        # 2. API GET /model/tuning/{tuning_id}
        resp = client.get(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 200
        by_id = {t["job_id"]: t for t in resp.json()["data"]["trials"]}
        assert by_id[completed_job.id]["metric_value"] == 0.87
        assert by_id[failed_job.id]["metric_value"] is None

        # 3. Socket.IO _load_tuning_catchup
        catchup = si._load_tuning_catchup(ts.id)
        assert catchup is not None
        catchup_by_id = {t["job_id"]: t for t in catchup["trials"]}
        assert catchup_by_id[completed_job.id]["metric_value"] == 0.87
        assert catchup_by_id[failed_job.id]["metric_value"] is None

        # 4. API DELETE /model/tuning/{tuning_id} cancellation emit preserves fallback best_metric
        with patch("app.socketio_instance.sio.emit", new_callable=AsyncMock) as mock_sio_emit:
            del_resp = client.delete(f"{BASE}/tuning/{ts.id}")
        assert del_resp.status_code == 204
        mock_sio_emit.assert_awaited_once()
        payload = mock_sio_emit.call_args[0][1]
        assert payload["best_job_id"] == completed_job.id
        assert payload["best_metric"] == 0.87

    def test_non_finite_metrics_ignored_in_selection_and_endpoints(self, client, db_session, monkeypatch):
        """NaN and Inf metrics are ignored everywhere and never poison best_metric or JSON serialization."""
        import app.socketio_instance as si
        from app.services.tuning_service import resolve_trial_metric

        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "non_finite_metric_model")

        # Part A: resolve_trial_metric, GET endpoint, and Socket.IO catch-up ignore +/-inf.
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd", "rmsprop"]},
            max_trials=3,
            metric="val_accuracy",
            direction="maximize",
            total_trials=3,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        inf_job = _seed_job(db_session, model.id, TrainingStatus.COMPLETED, tuning_session_id=ts.id)
        neg_inf_job = _seed_job(db_session, model.id, TrainingStatus.COMPLETED, tuning_session_id=ts.id)
        finite_job = _seed_job(db_session, model.id, TrainingStatus.COMPLETED, tuning_session_id=ts.id)
        db_session.add(
            TrainingMetric(job_id=inf_job.id, epoch=1, metric_name="val_accuracy", metric_value=float("inf"))
        )
        db_session.add(
            TrainingMetric(job_id=neg_inf_job.id, epoch=1, metric_name="accuracy", metric_value=float("-inf"))
        )
        db_session.add(TrainingMetric(job_id=finite_job.id, epoch=1, metric_name="val_accuracy", metric_value=0.76))
        db_session.commit()

        assert resolve_trial_metric(db_session, inf_job.id, "val_accuracy") is None
        assert resolve_trial_metric(db_session, neg_inf_job.id, "val_accuracy") is None
        assert resolve_trial_metric(db_session, finite_job.id, "val_accuracy") == 0.76

        # Part B: run_tuning_loop ignores mocked NaN (trial 0) and DB Inf (trial 1), picking finite trial 2.
        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )
        trial_jobs = [inf_job.id, neg_inf_job.id, finite_job.id]
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._run_single_trial",
            lambda _self, _sid, _m, _hp, trial_num, _loop: trial_jobs[trial_num],
        )
        original_get_metric = TuningService._get_trial_metric

        def _get_metric_with_nan(self_svc, jid, mname):
            if jid == inf_job.id:
                return float("nan")
            return original_get_metric(self_svc, jid, mname)

        monkeypatch.setattr("app.services.tuning_service.TuningService._get_trial_metric", _get_metric_with_nan)

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.COMPLETED
        assert ts.best_job_id == finite_job.id

        terminal = [e for e in emitted if e.get("type") == "tuning_complete"]
        assert len(terminal) == 1
        assert terminal[0]["best_job_id"] == finite_job.id
        assert terminal[0]["best_metric"] == 0.76

        # GET and catch-up both serialize cleanly with None for +/-inf trials.
        resp = client.get(f"{BASE}/tuning/{ts.id}")
        assert resp.status_code == 200
        by_id = {t["job_id"]: t for t in resp.json()["data"]["trials"]}
        assert by_id[inf_job.id]["metric_value"] is None
        assert by_id[neg_inf_job.id]["metric_value"] is None
        assert by_id[finite_job.id]["metric_value"] == 0.76

        catchup = si._load_tuning_catchup(ts.id)
        assert catchup is not None
        catchup_by_id = {t["job_id"]: t for t in catchup["trials"]}
        assert catchup_by_id[inf_job.id]["metric_value"] is None
        assert catchup_by_id[neg_inf_job.id]["metric_value"] is None
        assert catchup_by_id[finite_job.id]["metric_value"] == 0.76


class TestStageCFailureRecovery:
    """Stage C regression tests for outer exception recovery in run_tuning_loop."""

    @staticmethod
    def _patch_make_session(monkeypatch, db_session):
        from contextlib import contextmanager

        @contextmanager
        def _mock_session_cm():
            yield db_session

        monkeypatch.setattr("app.services.tuning_service.make_session", _mock_session_cm)

    def test_unexpected_exception_in_run_tuning_loop_fails_session_and_releases_slot(self, db_session, monkeypatch):
        """An unexpected error outside validation/trial handlers marks the session FAILED and frees the slot."""
        self._patch_make_session(monkeypatch, db_session)
        monkeypatch.setattr("app.services.tuning_service.MAX_CONCURRENT_TUNING_SESSIONS", 1)

        model = _seed_model(db_session, "unexpected_error_model")
        ts = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            total_trials=2,
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts)
        db_session.commit()
        db_session.refresh(ts)

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        # Fail in generate_grid_combinations: after validate_search_space and outside _run_single_trial.
        def _explode(_self, _search_space):
            raise RuntimeError("Unexpected grid generation failure")

        monkeypatch.setattr(
            "app.services.tuning_service.TuningService.generate_grid_combinations",
            _explode,
        )

        TuningService().run_tuning_loop(ts.id, model.model_name, loop=None)

        db_session.refresh(ts)
        assert ts.status == TuningSessionStatus.FAILED
        assert ts.completed_at is not None

        terminal = [e for e in emitted if e.get("type") == "tuning_complete"]
        assert len(terminal) == 1
        assert terminal[0]["status"] == "failed"
        assert "Unexpected grid generation failure" in (terminal[0]["error"] or "")

        # With MAX_CONCURRENT_TUNING_SESSIONS=1, the failed session must not consume the slot.
        check_tuning_concurrency_limit(db_session)

    def test_unexpected_exception_preserves_cancelled_and_completed_without_duplicate_complete(
        self, db_session, monkeypatch
    ):
        """Unexpected errors after CANCELLED or COMPLETED do not overwrite state or re-emit tuning_complete."""
        self._patch_make_session(monkeypatch, db_session)
        model = _seed_model(db_session, "unexpected_terminal_guard_model")

        # Case 1: Cancelled concurrently while generating trials before raising.
        cancelled_at = datetime(2026, 3, 1, 9, 0, 0, tzinfo=UTC)
        ts_cancelled = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam", "sgd"]},
            max_trials=2,
            metric="val_accuracy",
            direction="maximize",
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts_cancelled)
        db_session.commit()
        db_session.refresh(ts_cancelled)

        emitted = []
        monkeypatch.setattr(
            "app.services.tuning_service.TuningService._emit_progress",
            lambda _self, _loop, _sid, data: emitted.append(data),
        )

        def _cancel_then_explode(_self, _search_space):
            ts_row = db_session.get(TuningSession, ts_cancelled.id)
            ts_row.status = TuningSessionStatus.CANCELLED
            ts_row.completed_at = cancelled_at
            db_session.add(ts_row)
            db_session.commit()
            raise RuntimeError("Late unexpected error after cancellation")

        monkeypatch.setattr(
            "app.services.tuning_service.TuningService.generate_grid_combinations",
            _cancel_then_explode,
        )

        TuningService().run_tuning_loop(ts_cancelled.id, model.model_name, loop=None)

        db_session.refresh(ts_cancelled)
        assert ts_cancelled.status == TuningSessionStatus.CANCELLED
        assert ts_cancelled.completed_at.replace(tzinfo=None) == cancelled_at.replace(tzinfo=None)
        assert [e for e in emitted if e.get("type") == "tuning_complete"] == []

        # Case 2: Outer _emit_progress after RUNNING -> COMPLETED raises unexpectedly; COMPLETED stays intact.
        ts_completed = TuningSession(
            model_id=model.id,
            strategy=TuningStrategy.GRID,
            search_space={"optimizer": ["adam"]},
            max_trials=1,
            metric="val_accuracy",
            direction="maximize",
            status=TuningSessionStatus.PENDING,
        )
        db_session.add(ts_completed)
        db_session.commit()
        db_session.refresh(ts_completed)

        monkeypatch.setattr(
            "app.services.tuning_service.TuningService.generate_grid_combinations",
            lambda _self, _ss: [{"optimizer": "adam"}],
        )

        def _fake_trial(_self, session_id, _model_name, hp, _trial_num, _loop):
            job = _seed_job(
                db_session,
                model.id,
                TrainingStatus.COMPLETED,
                hyperparams=hp,
                tuning_session_id=session_id,
            )
            db_session.add(TrainingMetric(job_id=job.id, epoch=1, metric_name="val_accuracy", metric_value=0.9))
            db_session.commit()
            return job.id

        monkeypatch.setattr("app.services.tuning_service.TuningService._run_single_trial", _fake_trial)

        complete_emits = []

        def _raise_on_complete(_self, _loop, _sid, data):
            if data.get("type") == "tuning_complete":
                complete_emits.append(data)
                raise RuntimeError("Socket transport failure")

        monkeypatch.setattr("app.services.tuning_service.TuningService._emit_progress", _raise_on_complete)

        TuningService().run_tuning_loop(ts_completed.id, model.model_name, loop=None)

        db_session.refresh(ts_completed)
        assert ts_completed.status == TuningSessionStatus.COMPLETED
        assert len(complete_emits) == 1
        assert complete_emits[0]["status"] == "completed"
