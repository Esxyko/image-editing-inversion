"""Run pipeline parameter group 2 sequentially with every inversion method."""

import logging
from pathlib import Path
import subprocess
import sys
from time import perf_counter


PARAMETER_FILES = (
    "high_editing_guidance.yaml",
    "low_editing_guidance.yaml",
    "high_cross_replace.yaml",
)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    script_id = Path(__file__).stem
    log_path = project_root / "data" / "output" / "script-logs" / f"{script_id}.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not open script log {log_path}: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), file_handler],
        force=True,
    )
    logger = logging.getLogger(f"diffusion-script-{script_id}")
    total = len(PARAMETER_FILES)
    try:
        logger.info("Starting script %s: %d configurations; log=%s", script_id, total, log_path)
        for index, filename in enumerate(PARAMETER_FILES, start=1):
            started = perf_counter()
            logger.info("[%d/%d] Starting %s", index, total, filename)
            try:
                subprocess.run(
                    ["uv", "run", "diffuse", "--method", "ALL", "--h-params", filename],
                    cwd=project_root,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "[%d/%d] Failed %s: exit_code=%d; elapsed=%.2fs",
                    index, total, filename, exc.returncode, perf_counter() - started,
                )
                return exc.returncode if exc.returncode > 0 else 1
            except OSError as exc:
                logger.error(
                    "[%d/%d] Could not start %s: %s; elapsed=%.2fs",
                    index, total, filename, exc, perf_counter() - started,
                )
                return 1
            except KeyboardInterrupt:
                logger.error(
                    "[%d/%d] Interrupted %s by user; elapsed=%.2fs",
                    index, total, filename, perf_counter() - started,
                )
                return 130
            logger.info(
                "[%d/%d] Completed %s: exit_code=0; elapsed=%.2fs",
                index, total, filename, perf_counter() - started,
            )
        logger.info("Script %s completed all %d configurations", script_id, total)
        return 0
    finally:
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
