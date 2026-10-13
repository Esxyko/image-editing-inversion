"""Verify and publish complete run archives before removing their source directories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import stat
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from zipfile import ZIP_DEFLATED, ZipFile

from .._project import project_paths


@dataclass(frozen=True, slots=True)
class _RunEntry:
    relative_path: Path
    is_directory: bool
    size: int
    modified_ns: int
    changed_ns: int
    device: int
    inode: int


class RunArchiver:
    """Archive one finished, inactive run using the project's fixed output root.

    Callers must ensure no other process modifies the run or its archive during
    this operation. Snapshot checks detect changes but do not replace a writer lock.
    """

    def __init__(self, run_id: str) -> None:
        if (
            not isinstance(run_id, str)
            or re.fullmatch(r'[^\x00-\x1f<>:"/\\|?*]+', run_id) is None
            or run_id.endswith((".", " "))
        ):
            raise ValueError("run_id must be a single directory name, not a path")
        self._output = project_paths().output.resolve()
        self._source = self._output / run_id
        self._archive = self._output / f"{run_id}.zip"

    def _validate_source(self) -> None:
        if self._source.is_symlink() or self._source.is_junction():
            raise ValueError(f"Run directories must not be links or junctions: {self._source}")
        resolved = self._source.resolve(strict=True)
        if resolved != self._source or resolved.parent != self._output:
            raise ValueError(f"Run must be a direct child of {self._output}")
        if not stat.S_ISDIR(self._source.lstat().st_mode):
            raise ValueError(f"Run must be a directory: {self._source}")

    def _validate_destination(self) -> None:
        if self._archive.is_symlink() or self._archive.is_junction():
            raise ValueError(f"Archive destination must not be a link or junction: {self._archive}")
        try:
            status = self._archive.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"Archive destination must be a regular file: {self._archive}")

    def _snapshot(self) -> tuple[_RunEntry, ...]:
        self._validate_source()
        pending = [self._source]
        entries: list[_RunEntry] = []
        while pending:
            path = pending.pop()
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or path.is_junction():
                raise ValueError(f"Run contents must not contain links or junctions: {path}")
            is_directory = stat.S_ISDIR(status.st_mode)
            if not is_directory and not stat.S_ISREG(status.st_mode):
                raise ValueError(f"Run contents must be regular files or directories: {path}")
            entries.append(_RunEntry(
                relative_path=path.relative_to(self._source),
                is_directory=is_directory,
                size=0 if is_directory else status.st_size,
                modified_ns=status.st_mtime_ns,
                changed_ns=status.st_ctime_ns,
                device=status.st_dev,
                inode=status.st_ino,
            ))
            if is_directory:
                pending.extend(path.iterdir())
        return tuple(sorted(entries, key=lambda entry: entry.relative_path.as_posix()))

    def _require_finished(self) -> None:
        manifest_path = self._source / "sweep.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Run requires a readable, valid sweep.json: {manifest_path}") from exc
        if not isinstance(manifest, dict) or manifest.get("status") not in ("ok", "error"):
            raise ValueError(f"Run must have sweep.json status 'ok' or 'error': {self._source}")

    def _member_name(self, entry: _RunEntry) -> str:
        name = (Path(self._source.name) / entry.relative_path).as_posix()
        return f"{name}/" if entry.is_directory else name

    def _write_archive(self, stream: BinaryIO, entries: tuple[_RunEntry, ...]) -> None:
        with ZipFile(
            stream, "w", compression=ZIP_DEFLATED, compresslevel=6,
            allowZip64=True, strict_timestamps=False,
        ) as archive:
            for entry in entries:
                archive.write(
                    self._source / entry.relative_path,
                    arcname=self._member_name(entry),
                )

    def _verify_archive(self, path: Path, entries: tuple[_RunEntry, ...]) -> None:
        expected = {
            self._member_name(entry): (entry.size, entry.is_directory) for entry in entries
        }
        with ZipFile(path, "r", allowZip64=True) as archive:
            members = archive.infolist()
            actual = {
                member.filename: (member.file_size, member.is_dir()) for member in members
            }
            if len(members) != len(entries) or actual != expected:
                raise RuntimeError(f"ZIP members or sizes do not match the run: {path}")
            damaged = archive.testzip()
            if damaged is not None:
                raise RuntimeError(f"ZIP CRC verification failed for {damaged!r}: {path}")

    def _assert_unchanged(self, entries: tuple[_RunEntry, ...]) -> None:
        if self._snapshot() != entries:
            raise RuntimeError(f"Run changed during compression: {self._source}")
        self._require_finished()

    def compress(self) -> Path:
        """Return the published ZIP only after the original run has been removed."""
        entries = self._snapshot()
        self._require_finished()
        self._validate_destination()
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w+b", prefix=f".{self._source.name}-", suffix=".zip.tmp",
                dir=self._output, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                self._write_archive(stream, entries)
            self._verify_archive(temporary, entries)
            self._assert_unchanged(entries)
            self._validate_destination()
            temporary.replace(self._archive)
            temporary = None
        except BaseException as exc:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as cleanup:
                    exc.add_note(f"Temporary archive cleanup failed at {temporary}: {cleanup}")
            raise

        try:
            self._assert_unchanged(entries)
            shutil.rmtree(self._source)
        except BaseException as exc:
            message = (
                f"Verified ZIP saved at {self._archive}, but source cleanup did not complete; "
                f"the run directory may have been partially deleted: {self._source}"
            )
            if isinstance(exc, Exception):
                raise RuntimeError(message) from exc
            exc.add_note(message)
            raise
        return self._archive


def compress_run(run_id: str) -> Path:
    """Compress an inactive ok/error run, replace its sibling ZIP, and delete the run.

    Accept only the directory name under data/output/, independent of cwd. The
    ZIP preserves the run's top-level directory, all regular files, and empty
    directories. Links, junctions, special files, and unfinished runs are rejected.
    Publication follows member/size and full CRC verification. Earlier failures
    preserve the source and any old ZIP; cleanup failures retain the published ZIP
    and report its path, but may leave a partially deleted source directory.
    """
    return RunArchiver(run_id).compress()
