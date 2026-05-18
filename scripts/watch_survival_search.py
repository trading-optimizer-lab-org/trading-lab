from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ISSUE_TITLE = "Survival Search autonomous watchdog"
RESULT_ISSUE_TITLE = "Survival Search latest result"
DEFAULT_WORKFLOW = "survival-search.yml"
ACTIVE_STATUSES = {"queued", "in_progress", "waiting", "pending"}
RELAUNCHABLE_CONCLUSIONS = {"failure", "timed_out", "cancelled", "startup_failure"}


@dataclass(frozen=True)
class WatchDecision:
    status: str
    action: str
    body: str


class GitHubClient:
    def __init__(self, *, token: str, repository: str, api_url: str = "https://api.github.com") -> None:
        self.token = token
        self.repository = repository
        self.api_url = api_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.api_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": accept,
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "trading-lab-survival-watchdog",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                if not raw:
                    return None
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return json.loads(raw.decode("utf-8"))
                return raw
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API error {exc.code} for {method} {path}: {detail}") from exc

    def list_workflow_runs(self, workflow: str) -> list[dict[str, Any]]:
        data = self.request(
            "GET",
            f"/repos/{self.repository}/actions/workflows/{workflow}/runs?per_page=30",
        )
        return list(data.get("workflow_runs", []))

    def list_jobs(self, run_id: int) -> list[dict[str, Any]]:
        data = self.request("GET", f"/repos/{self.repository}/actions/runs/{run_id}/jobs?per_page=100")
        return list(data.get("jobs", []))

    def list_artifacts(self, run_id: int) -> list[dict[str, Any]]:
        data = self.request("GET", f"/repos/{self.repository}/actions/runs/{run_id}/artifacts?per_page=100")
        return list(data.get("artifacts", []))

    def download_artifact_zip(self, artifact_id: int) -> bytes:
        return self.request(
            "GET",
            f"/repos/{self.repository}/actions/artifacts/{artifact_id}/zip",
            accept="application/zip",
        )

    def list_open_issues(self) -> list[dict[str, Any]]:
        data = self.request("GET", f"/repos/{self.repository}/issues?state=open&per_page=100")
        return list(data)

    def create_issue(self, *, title: str, body: str) -> None:
        self.request("POST", f"/repos/{self.repository}/issues", payload={"title": title, "body": body})

    def update_issue(self, *, number: int, body: str) -> None:
        self.request("PATCH", f"/repos/{self.repository}/issues/{number}", payload={"body": body})

    def dispatch_workflow(self, workflow: str, *, ref: str = "main") -> None:
        self.request(
            "POST",
            f"/repos/{self.repository}/actions/workflows/{workflow}/dispatches",
            payload={"ref": ref},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous GitHub watchdog for Survival Search.")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-url-base", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF_NAME", "main"))
    parser.add_argument("--relaunch-on-terminal-problem", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    if not args.repo:
        raise SystemExit("GITHUB_REPOSITORY or --repo is required")

    client = GitHubClient(token=token, repository=args.repo)
    decision = watch_survival_search(
        client,
        workflow=args.workflow,
        repo=args.repo,
        run_url_base=args.run_url_base,
        ref=args.ref,
        relaunch_on_terminal_problem=args.relaunch_on_terminal_problem,
    )
    print(json.dumps({"status": decision.status, "action": decision.action}, indent=2))
    return 0


def watch_survival_search(
    client: GitHubClient,
    *,
    workflow: str,
    repo: str,
    run_url_base: str,
    ref: str,
    relaunch_on_terminal_problem: bool,
) -> WatchDecision:
    issues = client.list_open_issues()
    runs = client.list_workflow_runs(workflow)
    if not runs:
        body = _body_header(repo, run_url_base, None) + "\nNo hay runs de Survival Search todavia."
        _upsert_issue(client, issues, ISSUE_TITLE, body)
        return WatchDecision(status="no_runs", action="issue_updated", body=body)

    active_runs = [run for run in runs if str(run.get("status")) in ACTIVE_STATUSES]
    run = active_runs[0] if active_runs else runs[0]
    run_id = int(run["id"])
    jobs = client.list_jobs(run_id)
    summary = _read_survival_summary(client, run_id) if _is_success(run) else None
    body = _build_watch_body(
        repo=repo,
        run_url_base=run_url_base,
        run=run,
        jobs=jobs,
        summary=summary,
        relaunch_marker=None,
    )

    action = "issue_updated"
    if (
        relaunch_on_terminal_problem
        and not active_runs
        and _is_relaunchable(run)
        and not _already_relaunched(issues, run_id)
    ):
        client.dispatch_workflow(workflow, ref=ref)
        action = "relaunched"
        body = _build_watch_body(
            repo=repo,
            run_url_base=run_url_base,
            run=run,
            jobs=jobs,
            summary=summary,
            relaunch_marker=f"Relaunched from failed run: {run_id}",
        )

    _upsert_issue(client, issues, ISSUE_TITLE, body)
    return WatchDecision(status=str(run.get("status", "")), action=action, body=body)


def _read_survival_summary(client: GitHubClient, run_id: int) -> dict[str, Any] | None:
    artifacts = client.list_artifacts(run_id)
    artifact = next((item for item in artifacts if item.get("name") == "survival-leaderboard"), None)
    if not artifact:
        return None
    raw = client.download_artifact_zip(int(artifact["id"]))
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = next((item for item in archive.namelist() if item.endswith("survival_summary.json")), None)
        if not name:
            return None
        return json.loads(archive.read(name).decode("utf-8"))


def _build_watch_body(
    *,
    repo: str,
    run_url_base: str,
    run: dict[str, Any],
    jobs: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    relaunch_marker: str | None,
) -> str:
    run_id = int(run["id"])
    run_url = f"{run_url_base}/{repo}/actions/runs/{run_id}"
    status = str(run.get("status", "unknown"))
    conclusion = str(run.get("conclusion") or "none")
    completed = sum(1 for job in jobs if job.get("status") == "completed")
    failed_jobs = [
        str(job.get("name"))
        for job in jobs
        if job.get("status") == "completed" and job.get("conclusion") not in {"success", "skipped"}
    ]
    lines = [
        "# Survival Search autonomous watchdog",
        "",
        f"Run: {run_url}",
        f"Status: {status}",
        f"Conclusion: {conclusion}",
        f"Jobs completed: {completed} / {len(jobs)}",
        f"Failed jobs: {', '.join(failed_jobs) if failed_jobs else 'none'}",
    ]
    if summary:
        best = summary.get("best") or {}
        lines.extend(
            [
                "",
                "## Latest artifact summary",
                "",
                f"Candidates evaluated: {summary.get('candidates_evaluated', summary.get('rows', 0))}",
                f"Accepted: {summary.get('accepted', 0)}",
                f"Best ID: {best.get('candidate_id', 'none')}",
                f"Best rule: {best.get('rule', 'none')}",
                f"Best feature: {best.get('feature_name', 'n/a')}",
                f"Train Calmar: {best.get('train_calmar', 'n/a')}",
                f"Validation Calmar: {best.get('validation_calmar', 'n/a')}",
                f"Robust passes: {best.get('robust_passes', 'n/a')} / {best.get('robust_total', 'n/a')}",
                f"Accepted best: {best.get('accepted', False)}",
                f"Rejection reason: {best.get('rejection_reason', 'none')}",
            ]
        )
    else:
        lines.extend(["", "Artifact summary: not available yet."])
    if relaunch_marker:
        lines.extend(["", relaunch_marker])
    lines.extend(["", "_Updated automatically by GitHub Actions. No OpenAI API key used._"])
    return "\n".join(lines)


def _body_header(repo: str, run_url_base: str, run_id: int | None) -> str:
    run_line = "Run: none" if run_id is None else f"Run: {run_url_base}/{repo}/actions/runs/{run_id}"
    return "\n".join(["# Survival Search autonomous watchdog", "", run_line])


def _upsert_issue(client: GitHubClient, issues: list[dict[str, Any]], title: str, body: str) -> None:
    issue = next((item for item in issues if item.get("title") == title), None)
    if issue:
        client.update_issue(number=int(issue["number"]), body=body)
    else:
        client.create_issue(title=title, body=body)


def _is_success(run: dict[str, Any]) -> bool:
    return run.get("status") == "completed" and run.get("conclusion") == "success"


def _is_relaunchable(run: dict[str, Any]) -> bool:
    return run.get("status") == "completed" and str(run.get("conclusion")) in RELAUNCHABLE_CONCLUSIONS


def _already_relaunched(issues: list[dict[str, Any]], run_id: int) -> bool:
    marker = f"Relaunched from failed run: {run_id}"
    return any(marker in str(issue.get("body", "")) for issue in issues)


if __name__ == "__main__":
    raise SystemExit(main())
