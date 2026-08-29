#!/usr/bin/env python3
"""Download replay artifacts for the top Kaggriculture leaderboard entries.

This intentionally shells out to the official Kaggle CLI instead of calling
private Kaggle endpoints.  It downloads serially, pauses between requests, and
can be safely resumed by rerunning it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse


COMPETITION = "kaggriculture"
DEFAULT_DELAY = 2.0
ID_FIELDS = ("submissionId", "submission_id", "teamId", "team_id")


def ids_from_url(value: str) -> tuple[str | None, str | None]:
    query = parse_qs(urlparse(value).query)
    submission = query.get("submissionId", [])
    episode = query.get("episodeId", [])
    submission_id = submission[0] if submission and submission[0].isdigit() else None
    episode_id = episode[0] if episode and episode[0].isdigit() else None
    return submission_id, episode_id


def run_kaggle(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["kaggle", *args]
    environment = os.environ.copy()
    if "KAGGLE_API_TOKEN" in environment:
        environment["KAGGLE_API_TOKEN"] = environment["KAGGLE_API_TOKEN"].strip()
    return subprocess.run(command, check=True, text=True, capture_output=capture, env=environment)


def download_leaderboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    run_kaggle(
        ["competitions", "leaderboard", COMPETITION, "--download", "--path", str(path.parent), "--quiet"]
    )
    downloaded = path.parent / f"{COMPETITION}.csv"
    archive = path.parent / f"{COMPETITION}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            member = next((name for name in bundle.namelist() if name.endswith(".csv")), None)
            if member:
                path.write_bytes(bundle.read(member))
    elif downloaded != path and downloaded.exists():
        downloaded.replace(path)
    if not path.exists():
        raise RuntimeError(f"Kaggle CLI did not create {path}")


def leaderboard_teams(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        normalized = {field.lower(): field for field in rows.fieldnames or []}
        id_fields = [normalized[name.lower()] for name in ID_FIELDS if name.lower() in normalized]
        teams: list[tuple[str, str]] = []
        for row in rows:
            value = next((row.get(field, "") for field in id_fields if row.get(field)), "")
            match = re.search(r"\d+", value)
            if match and (match.group(), row.get("Rank", "")) not in teams:
                teams.append((row.get("Rank", ""), match.group()))
    if not teams:
        columns = ", ".join(rows.fieldnames or [])
        raise RuntimeError(
            f"{path} has no submission IDs (columns: {columns}). "
            "The current Kaggle CLI may return teamId; the caller must resolve teams to submissions."
        )
    return teams


def json_output(args: list[str]) -> object:
    result = run_kaggle(args + ["--format", "json", "--quiet"], capture=True)
    return json.loads(result.stdout)


def latest_episode_id(submission_id: str) -> str | None:
    episodes = json_output(["competitions", "episodes", submission_id])
    if not isinstance(episodes, list):
        return None
    for episode in episodes:
        if isinstance(episode, dict) and episode.get("id"):
            return str(episode["id"])
    return None


def replay_file(directory: Path, episode_id: str) -> Path:
    return directory / f"episode-{episode_id}-replay.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-url", action="append", default=[], help="Optional Kaggle leaderboard URL")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between downloads")
    parser.add_argument("--output", type=Path, default=Path("kaggle_replays"))
    parser.add_argument("--leaderboard", type=Path, default=Path("kaggle_leaderboard.csv"))
    args = parser.parse_args()
    if args.limit < 1 or args.delay < 0:
        parser.error("--limit must be positive and --delay cannot be negative")

    try:
        explicit_ids = [ids_from_url(url) for url in args.leaderboard_url]
        replay_ids: list[tuple[str, str, str]] = []
        for submission_id, episode_id in explicit_ids:
            if episode_id:
                replay_ids.append(("", "", submission_id or episode_id, episode_id))
            elif submission_id:
                time.sleep(args.delay)
                resolved_episode = latest_episode_id(submission_id)
                if resolved_episode:
                    replay_ids.append(("", "", submission_id, resolved_episode))
        if not replay_ids:
            download_leaderboard(args.leaderboard)
            teams = leaderboard_teams(args.leaderboard)
            for rank, team_id in teams:
                time.sleep(args.delay)
                submissions = json_output(["competitions", "team-submissions", team_id])
                if isinstance(submissions, list):
                    for submission in submissions:
                        if isinstance(submission, dict) and submission.get("id"):
                            submission_id = str(submission["id"])
                            time.sleep(args.delay)
                            episode_id = latest_episode_id(submission_id)
                            if episode_id:
                                pair = (rank, team_id, submission_id, episode_id)
                                if pair not in replay_ids:
                                    replay_ids.append(pair)
                            if len(replay_ids) >= args.limit:
                                break
                if len(replay_ids) >= args.limit:
                    break
            if not replay_ids:
                raise RuntimeError("No submissions found for the leaderboard teams")
        args.output.mkdir(parents=True, exist_ok=True)
        replay_ids = list(dict.fromkeys(replay_ids))[:args.limit]
        manifest = args.output / "replays.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rank", "team_id", "submission_id", "episode_id", "replay_path"])
            writer.writeheader()
            for rank, team_id, submission_id, episode_id in replay_ids:
                writer.writerow({"rank": rank, "team_id": team_id, "submission_id": submission_id, "episode_id": episode_id, "replay_path": str(replay_file(args.output / submission_id, episode_id))})
        for index, (rank, team_id, submission_id, episode_id) in enumerate(replay_ids, start=1):
            target = args.output / submission_id
            if replay_file(target, episode_id).is_file():
                print(f"[{index}/{len(replay_ids)}] already downloaded {submission_id}")
                continue
            target.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(replay_ids)}] downloading submission {submission_id}", flush=True)
            run_kaggle(["competitions", "replay", episode_id, "--path", str(target), "--quiet"])
            if index != len(replay_ids):
                time.sleep(args.delay)
    except FileNotFoundError:
        print("kaggle CLI not found; install and authenticate the official Kaggle CLI first.", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"Kaggle CLI failed with exit code {error.returncode}: {error.cmd}", file=sys.stderr)
        return error.returncode or 1
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
