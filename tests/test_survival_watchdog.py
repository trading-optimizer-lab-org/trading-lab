from __future__ import annotations

import io
import json
import zipfile

from scripts.watch_survival_search import ISSUE_TITLE, watch_survival_search


class FakeClient:
    def __init__(self, *, runs, jobs=None, issues=None, artifacts=None, artifact_summary=None):
        self.runs = runs
        self.jobs = jobs or []
        self.issues = issues or []
        self.artifacts = artifacts or []
        self.artifact_summary = artifact_summary
        self.created = []
        self.updated = []
        self.dispatched = []

    def list_open_issues(self):
        return self.issues

    def list_workflow_runs(self, workflow):
        return self.runs

    def list_jobs(self, run_id):
        return self.jobs

    def list_artifacts(self, run_id):
        return self.artifacts

    def download_artifact_zip(self, artifact_id):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("survival_summary.json", json.dumps(self.artifact_summary))
        return buffer.getvalue()

    def create_issue(self, *, title, body):
        self.created.append({"title": title, "body": body})

    def update_issue(self, *, number, body):
        self.updated.append({"number": number, "body": body})

    def dispatch_workflow(self, workflow, *, ref="main"):
        self.dispatched.append({"workflow": workflow, "ref": ref})


def test_watchdog_updates_issue_for_active_run() -> None:
    client = FakeClient(
        runs=[{"id": 123, "status": "in_progress", "conclusion": None}],
        jobs=[{"name": "data", "status": "completed", "conclusion": "success"}],
    )

    decision = watch_survival_search(
        client,
        workflow="survival-search.yml",
        repo="owner/repo",
        run_url_base="https://github.com",
        ref="main",
        relaunch_on_terminal_problem=True,
    )

    assert decision.action == "issue_updated"
    assert client.created[0]["title"] == ISSUE_TITLE
    assert "Status: in_progress" in client.created[0]["body"]
    assert not client.dispatched


def test_watchdog_reads_success_artifact_summary() -> None:
    client = FakeClient(
        runs=[{"id": 456, "status": "completed", "conclusion": "success"}],
        jobs=[{"name": "merge", "status": "completed", "conclusion": "success"}],
        artifacts=[{"id": 99, "name": "survival-leaderboard"}],
        artifact_summary={
            "candidates_evaluated": 10,
            "accepted": 1,
            "best": {
                "candidate_id": "abc",
                "rule": "feature_threshold",
                "feature_name": "vix_term",
                "train_calmar": 1.3,
                "validation_calmar": 1.4,
                "robust_passes": 14,
                "robust_total": 14,
                "accepted": True,
            },
        },
    )

    decision = watch_survival_search(
        client,
        workflow="survival-search.yml",
        repo="owner/repo",
        run_url_base="https://github.com",
        ref="main",
        relaunch_on_terminal_problem=True,
    )

    assert decision.action == "issue_updated"
    assert "Candidates evaluated: 10" in decision.body
    assert "Best ID: abc" in decision.body
    assert "Validation Calmar: 1.4" in decision.body


def test_watchdog_relaunches_failed_run_once() -> None:
    client = FakeClient(
        runs=[{"id": 789, "status": "completed", "conclusion": "failure"}],
        jobs=[{"name": "survival (1)", "status": "completed", "conclusion": "failure"}],
        issues=[],
    )

    decision = watch_survival_search(
        client,
        workflow="survival-search.yml",
        repo="owner/repo",
        run_url_base="https://github.com",
        ref="main",
        relaunch_on_terminal_problem=True,
    )

    assert decision.action == "relaunched"
    assert client.dispatched == [{"workflow": "survival-search.yml", "ref": "main"}]
    assert "Relaunched from failed run: 789" in decision.body


def test_watchdog_does_not_relaunch_same_failed_run_twice() -> None:
    client = FakeClient(
        runs=[{"id": 789, "status": "completed", "conclusion": "failure"}],
        issues=[{"number": 1, "title": ISSUE_TITLE, "body": "Relaunched from failed run: 789"}],
    )

    decision = watch_survival_search(
        client,
        workflow="survival-search.yml",
        repo="owner/repo",
        run_url_base="https://github.com",
        ref="main",
        relaunch_on_terminal_problem=True,
    )

    assert decision.action == "issue_updated"
    assert not client.dispatched
    assert client.updated
