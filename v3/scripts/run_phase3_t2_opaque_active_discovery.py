"""Run the source-bound, disk-staged Phase III-T2 evidence protocol.

Every command fails closed until a separately frozen, execution-ready protocol
binds fresh controller entropy, all roots, the exact ten-arm schedule, and
source/runtime closure.  The official workflow is:

1. ``preopen-environment --index`` independently persists each answer-free
   preopen summary;
2. ``aggregate-preopen`` binds the exact ten summaries in schedule order;
3. the sole ``open`` command reconstructs all ten summaries and the terminal
   before atomically opening any inferred, sealed, or long-program sidecar;
4. ``validate`` performs the same authoritative reconstruction against the
   persisted report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tnlm_v3.opaque_active_discovery_protocol import (
    T2ProtocolError,
    aggregate_t2_preopen_records,
    load_phase3_t2_protocol,
    load_t2_campaign_report,
    load_t2_campaign_terminal_preopen,
    load_t2_preopen_record_set,
    open_t2_staged_campaign,
    reconstruct_t2_preopen_record,
    require_execution_ready,
    t2_preopen_record_path,
    validate_t2_staged_campaign,
    write_t2_campaign_report,
    write_t2_campaign_terminal_preopen,
    write_t2_preopen_record,
)


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "phase3" / "opaque_active_discovery_t2.json"


def default_evidence_directory() -> Path:
    return default_repository_root() / "v3_recovery" / "phase3_t2_opaque_active_discovery"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=default_protocol_path())
    parser.add_argument("--repository-root", type=Path, default=default_repository_root())
    parser.add_argument("--evidence-dir", type=Path, default=default_evidence_directory())


def _terminal_path(evidence_directory: Path) -> Path:
    return evidence_directory / "terminal-preopen.json"


def _report_path(evidence_directory: Path) -> Path:
    return evidence_directory / "campaign-report.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    preopen = subcommands.add_parser(
        "preopen-environment",
        help="run and atomically persist one scheduled answer-free preopen summary",
    )
    _add_common_arguments(preopen)
    preopen.add_argument("--index", type=int, required=True, choices=range(10))

    aggregate = subcommands.add_parser(
        "aggregate-preopen",
        help="freeze the exact ten-record schedule into its terminal",
    )
    _add_common_arguments(aggregate)

    open_parser = subcommands.add_parser(
        "open",
        help="perform the sole atomic all-environment postfit opening",
    )
    _add_common_arguments(open_parser)

    validate = subcommands.add_parser(
        "validate",
        help="authoritatively reconstruct and validate the staged report",
    )
    _add_common_arguments(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        protocol = load_phase3_t2_protocol(args.protocol)
        evidence_directory = args.evidence_dir.resolve()
        repository_root = args.repository_root.resolve()
        require_execution_ready(protocol, repository_root)

        if args.command == "preopen-environment":
            record = reconstruct_t2_preopen_record(protocol, repository_root, args.index)
            destination = t2_preopen_record_path(evidence_directory, args.index)
            write_t2_preopen_record(destination, record)
            print(f"Phase III-T2 preopen {args.index:02d} persisted: {record.preopen_sha256}")
            return 0

        records = load_t2_preopen_record_set(evidence_directory)
        if args.command == "aggregate-preopen":
            terminal = aggregate_t2_preopen_records(protocol, repository_root, records)
            terminal_path = _terminal_path(evidence_directory)
            if terminal_path.exists():
                existing_terminal = load_t2_campaign_terminal_preopen(terminal_path)
                if existing_terminal != terminal:
                    raise T2ProtocolError("existing terminal is not byte-semantically identical")
            write_t2_campaign_terminal_preopen(terminal_path, terminal)
            print(f"Phase III-T2 terminal persisted: {terminal.terminal_sha256}")
            return 0

        terminal = load_t2_campaign_terminal_preopen(_terminal_path(evidence_directory))
        if args.command == "open":
            report_path = _report_path(evidence_directory)
            if report_path.exists():
                existing_report = load_t2_campaign_report(report_path)
                validate_t2_staged_campaign(
                    protocol,
                    repository_root,
                    records,
                    terminal,
                    existing_report,
                )
                write_t2_campaign_report(report_path, existing_report)
                print(f"Phase III-T2 campaign already atomically open: {existing_report.report_sha256}")
                return 0
            report = open_t2_staged_campaign(
                protocol,
                repository_root,
                records,
                terminal,
            )
            write_t2_campaign_report(report_path, report)
            print(f"Phase III-T2 campaign atomically opened: {report.report_sha256}")
            return 0

        if args.command == "validate":
            report = load_t2_campaign_report(_report_path(evidence_directory))
            validate_t2_staged_campaign(
                protocol,
                repository_root,
                records,
                terminal,
                report,
            )
            print(f"Phase III-T2 campaign validated: {report.report_sha256}")
            return 0
        raise AssertionError("argparse accepted an unknown command")
    except (OSError, TypeError, ValueError, T2ProtocolError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
