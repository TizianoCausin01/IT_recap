"""
Download a small, targeted slice of the Triple-N dataset over FTP.

The full release is about 3 TB, almost all of it raw session folders. Nothing
in this project needs those: the spike rasters live in the GoodUnit files and
the per-unit quality labels in the Processed files, so a single session costs a
few GB rather than tens. This script lists the remote share first and refuses
to start a transfer that would not fit in the free space that remains.

Science Data Bank serves the files over FTP and shows the host, port, and
credentials in the "Data File Download" panel of the dataset page once you are
logged in (doi:10.57760/sciencedb.33556). Pass them through the environment so
they never land in the shell history or in this file:

    export SCIDB_FTP_HOST=...
    export SCIDB_FTP_USER=...
    export SCIDB_FTP_PASSWORD=...
    python download_triple_n_sessions.py --list_only
    python download_triple_n_sessions.py --sessions 240629 --include_processed
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "python_scripts" / "src"
sys.path.insert(0, str(PROJECT_SRC))

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

from IT_recap.triple_n import (  # noqa: E402
    download_triple_n_files,
    list_triple_n_remote_dir,
)


# Documented layout of the Science Data Bank share.
GOOD_UNIT_DIR = "Raw/GoodStruct"
PROCESSED_DIR = "Processed"


@dataclass
class Cfg:
    # Environment and destination. None resolves through config.yaml.
    env: str = ENV
    output_dir: str | None = None

    # FTP credentials, read from the environment by default.
    host: str | None = None
    username: str | None = None
    password: str | None = None
    port: int = 21

    # Selection. Sessions are matched by the YYMMDD stamp in the file name.
    sessions: str = ""
    good_unit_dir: str = GOOD_UNIT_DIR
    processed_dir: str = PROCESSED_DIR
    include_processed: bool = True

    # Safety rails around a nearly full disk.
    list_only: bool = False
    reserve_gb: float = 5.0
    overwrite: bool = False


"""
parse_args
Parse command-line overrides into the Triple-N download configuration.

OUTPUT:
    - cfg: Cfg -> credentials, session selection, and disk-safety settings
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected Triple-N GoodUnit and Processed files from the "
            "Science Data Bank FTP share."
        )
    )
    parser.add_argument("--env", default=Cfg.env, choices=config)
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--host", default=os.getenv("SCIDB_FTP_HOST"))
    parser.add_argument("--username", default=os.getenv("SCIDB_FTP_USER"))
    parser.add_argument("--password", default=os.getenv("SCIDB_FTP_PASSWORD"))
    parser.add_argument("--port", type=int, default=Cfg.port)
    parser.add_argument(
        "--sessions",
        default=Cfg.sessions,
        help="Comma-separated YYMMDD session stamps, e.g. 240629,240701.",
    )
    parser.add_argument("--good_unit_dir", default=Cfg.good_unit_dir)
    parser.add_argument("--processed_dir", default=Cfg.processed_dir)
    parser.add_argument(
        "--no_processed", dest="include_processed", action="store_false"
    )
    parser.add_argument("--list_only", action="store_true")
    parser.add_argument("--reserve_gb", type=float, default=Cfg.reserve_gb)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(include_processed=Cfg.include_processed)
    return Cfg(**vars(parser.parse_args()))
# EOF


"""
select_session_files
Keep the remote entries whose names carry one of the requested session stamps.

INPUT:
    - entries: list[tuple[str, int]] -> (file name, size) pairs from a listing
    - session_stamps: list[str] -> YYMMDD stamps; empty keeps everything

OUTPUT:
    - selected: list[tuple[str, int]] -> matching entries
"""
def select_session_files(entries, session_stamps):
    if not session_stamps:
        return list(entries)
    # end if no session filter was requested
    return [
        (file_name, file_size)
        for file_name, file_size in entries
        if any(stamp in file_name for stamp in session_stamps)
    ]
# EOF


"""
print_listing
Show a remote directory listing with sizes, smallest first.

INPUT:
    - title: str -> heading for this listing
    - entries: list[tuple[str, int]] -> (file name, size) pairs

OUTPUT:
    - None
"""
def print_listing(title, entries):
    print(f"\n{title} ({len(entries)} entries)")
    for file_name, file_size in sorted(entries, key=lambda item: item[1]):
        size_note = "  (dir)" if file_size < 0 else f"{file_size / 1e9:9.2f} GB"
        print(f"  {size_note}  {file_name}")
    # end for listed entry
# EOF


def main():
    cfg = parse_args()
    if not (cfg.host and cfg.username and cfg.password):
        raise SystemExit(
            "Missing FTP credentials. Log in at https://www.scidb.cn/en/, open "
            "the dataset's 'Data File Download' panel, and export "
            "SCIDB_FTP_HOST, SCIDB_FTP_USER, and SCIDB_FTP_PASSWORD."
        )
    # end if credentials were not supplied

    paths = config[cfg.env]["paths"]
    output_dir = Path(
        cfg.output_dir or Path(paths["data_path"]) / "data" / "triple_n"
    ).expanduser()
    session_stamps = [
        stamp.strip() for stamp in cfg.sessions.split(",") if stamp.strip()
    ]

    remote_dirs = [(cfg.good_unit_dir, "GoodUnit")]
    if cfg.include_processed:
        remote_dirs.append((cfg.processed_dir, "Processed"))
    # end if per-unit summaries were requested

    planned = []
    for remote_dir, label in remote_dirs:
        entries = list_triple_n_remote_dir(
            cfg.host, cfg.username, cfg.password, remote_dir, cfg.port
        )
        selected = select_session_files(entries, session_stamps)
        print_listing(f"{label} @ {remote_dir}", entries if cfg.list_only else selected)
        planned.extend(
            (f"{remote_dir}/{file_name}", file_size)
            for file_name, file_size in selected
            if file_size > 0
        )
    # end for remote directory

    if cfg.list_only:
        print("\nListing only; nothing downloaded.")
        return
    # end if the user only wanted to see what is available
    if not planned:
        raise SystemExit(
            f"No files matched sessions {session_stamps or '(all)'}. Re-run "
            "with --list_only to see the available names."
        )
    # end if the selection is empty

    # Refuse to start a transfer that would leave the disk with no headroom;
    # a half-written multi-GB file on a full volume is the worst outcome here.
    required_bytes = sum(size for _, size in planned)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir).free
    reserve_bytes = cfg.reserve_gb * 1e9
    print(
        f"\nselected {len(planned)} files | {required_bytes / 1e9:.2f} GB needed "
        f"| {free_bytes / 1e9:.2f} GB free | {cfg.reserve_gb:.1f} GB reserved"
    )
    if required_bytes + reserve_bytes > free_bytes:
        raise SystemExit(
            "Not enough free space. Select fewer sessions, drop --include_"
            "processed, or lower --reserve_gb if you accept the risk."
        )
    # end if the download would fill the disk

    local_paths = download_triple_n_files(
        cfg.host,
        cfg.username,
        cfg.password,
        [remote_path for remote_path, _ in planned],
        output_dir,
        port=cfg.port,
        overwrite=cfg.overwrite,
    )
    print(f"\ndownloaded {len(local_paths)} files into {output_dir}")
# EOF


if __name__ == "__main__":
    main()
