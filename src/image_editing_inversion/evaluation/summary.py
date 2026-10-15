"""Extract existing report statistics without loading datasets or metric models."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
from uuid import uuid4

from .._project import project_paths
from .models import GROUPS, METRICS


def _get(document: dict[str, Any], *keys: str) -> Any:
    value: Any = document
    for index, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing field: {'.'.join(keys[:index + 1])}")
        value = value[key]
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field}: expected a nonempty object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected a nonempty string")
    return value


def _count(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field}: expected a nonnegative integer")
    return value


def _number(value: Any, field: str, *, nullable: bool = False, infinite: bool = False) -> Any:
    if (nullable and value is None) or (infinite and value == "+inf"):
        return value
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{field}: expected a finite number" + (" or null" if nullable else ""))
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if value == "+inf":
        return "+∞"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    ).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
    ])


class MetricsSummarizer:
    """Keep each input/parameter/method/group separate in two compact outputs."""

    def __init__(self, reports: Sequence[str | Path] | None = None) -> None:
        self.root = project_paths().metrics
        self.output_root = self.root / "summaries"
        if isinstance(reports, (str, bytes, Path)):
            raise TypeError("reports must be a sequence of JSON files or directories")
        candidates = []
        for value in reports if reports is not None else (self.root,):
            path = Path(value).expanduser().resolve(strict=True)
            if path.is_relative_to(self.output_root.resolve()):
                continue
            if path.is_dir():
                candidates.extend(sorted(
                    child.resolve() for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() == ".json"
                    and not child.resolve().is_relative_to(self.output_root.resolve())
                ))
            elif path.is_file() and path.suffix.lower() == ".json":
                candidates.append(path)
            else:
                raise ValueError(f"Expected a JSON report or directory: {path}")
        self.reports = tuple(dict.fromkeys(candidates))
        if not self.reports:
            raise ValueError("No JSON reports selected")

    @staticmethod
    def _settings(method: dict[str, Any], prefix: str) -> dict[str, Any]:
        config = _object(_get(method, "configuration", "config"), prefix + ".configuration.config")
        settings = {}
        for section, keys in (
            ("model", ("model_id", "revision", "width", "height")),
            ("sampling", ("num_inference_steps", "seed", "editing_guidance_scale")),
            ("prompt_to_prompt", ("cross_replace_fraction", "self_replace_fraction")),
        ):
            for key in keys:
                value = _get(config, section, key)
                field = f"{prefix}.configuration.config.{section}.{key}"
                if key == "revision":
                    if value is not None:
                        _text(value, field)
                elif key == "model_id":
                    _text(value, field)
                elif key == "seed":
                    if type(value) is not int:
                        raise ValueError(f"{field}: expected an integer")
                elif key in ("width", "height", "num_inference_steps"):
                    _count(value, field)
                else:
                    _number(value, field)
                settings[key] = value
        params = _object(_get(config, "method_h_params"), prefix + ".configuration.config.method_h_params")
        settings["method_h_params"] = _json(params)
        return settings

    @staticmethod
    def _group(group: dict[str, Any], prefix: str) -> dict[str, Any]:
        row = {"sample_count": _count(_get(group, "sample_count"), prefix + ".sample_count")}
        for metric in METRICS:
            path = f"{prefix}.metrics.{metric}"
            stats = _object(_get(group, "metrics", metric), path)
            n = _count(_get(stats, "n"), path + ".n")
            if n > row["sample_count"]:
                raise ValueError(f"{path}.n: exceeds sample_count")
            row[f"{metric}_n"] = n
            for key in ("mean", "std"):
                value = _number(_get(stats, key), path + "." + key, nullable=True,
                                infinite=metric == "masked_psnr_db" and key == "mean")
                if (n == 0 and value is not None) or (key == "mean" and n > 0 and value is None):
                    raise ValueError(f"{path}.{key}: inconsistent with n={n}")
                if key == "std" and value is not None and (value < 0 or n < 2):
                    raise ValueError(f"{path}.std: invalid sample standard deviation")
                row[f"{metric}_{key}"] = value
        for phase in ("inversion", "editing"):
            path = f"{prefix}.timing.{phase}"
            timing = _object(_get(group, "timing", phase), path)
            n = _count(_get(timing, "n"), path + ".n")
            if n > row["sample_count"]:
                raise ValueError(f"{path}.n: exceeds sample_count")
            row[f"{phase}_n"] = n
            for key in ("total_core_seconds", "mean_core_seconds_per_image"):
                value = _number(_get(timing, key), path + "." + key, nullable=True)
                if (n == 0) != (value is None) or (value is not None and value < 0):
                    raise ValueError(f"{path}.{key}: inconsistent with n={n} or negative")
                row[f"{phase}_{key}"] = value
        cache = _object(_get(group, "inversion_cache"), prefix + ".inversion_cache")
        for key in ("known_count", "hit_count"):
            row[f"inversion_cache_{key}"] = _count(_get(cache, key), f"{prefix}.inversion_cache.{key}")
        known, hits = row["inversion_cache_known_count"], row["inversion_cache_hit_count"]
        rate = _number(_get(cache, "hit_rate"), prefix + ".inversion_cache.hit_rate", nullable=True)
        if not hits <= known <= row["sample_count"]:
            raise ValueError(f"{prefix}.inversion_cache: invalid counts")
        if ((known == 0) != (rate is None)
                or (rate is not None and (not 0 <= rate <= 1 or not math.isclose(rate, hits / known)))):
            raise ValueError(f"{prefix}.inversion_cache.hit_rate: inconsistent with cache counts")
        row["inversion_cache_hit_rate"] = rate
        return row

    def _extract(self, document: dict[str, Any], source: Path, source_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if type(_get(document, "schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("schema_version: only version 1 is supported")
        if _get(document, "status") != "complete":
            raise ValueError("status: expected complete")
        metadata = {"source_id": source_id, "source_report": str(source),
                    "archive_filename": _text(_get(document, "archive", "filename"), "archive.filename")}
        for key in ("reference", "fingerprint", "content_digest", "split", "source_assignment"):
            metadata[f"dataset_{key}"] = _text(_get(document, "dataset", key), "dataset." + key)
        protocol = {}
        for key in ("clip", "edited_region_clip", "masked_background", "psnr", "ssim", "lpips",
                    "empty_region_policy", "aggregation", "timing_protocol"):
            value = _get(document, "evaluation", key)
            if key in ("edited_region_clip", "masked_background", "empty_region_policy", "aggregation"):
                _text(value, "evaluation." + key)
            else:
                _object(value, "evaluation." + key)
            protocol[key] = value
        # Keep protocol definitions, but omit CLIP's verbose processor configuration.
        protocol["clip"] = {key: _get(protocol["clip"], key) for key in (
            "model", "revision", "input_resolution", "precision", "tf32", "max_text_tokens",
            "text_overflow", "score",
        )}
        metadata["evaluation_protocol"] = _json(protocol)
        rows = []
        parameters = _object(_get(document, "parameter_files"), "parameter_files")
        for parameter_file, parameter in sorted(parameters.items()):
            prefix = f"parameter_files.{parameter_file}.methods"
            methods = _object(_get(parameter, "methods"), prefix)
            for method_id, method in sorted(methods.items()):
                method_prefix = prefix + "." + method_id
                _object(method, method_prefix)
                settings = self._settings(method, method_prefix)
                for name in GROUPS:
                    group = _object(_get(method, "groups", name), method_prefix + ".groups." + name)
                    rows.append({**metadata, "parameter_file": parameter_file, "method": method_id,
                                 **settings, "group": name,
                                 **self._group(group, method_prefix + ".groups." + name)})
        return metadata, rows

    @staticmethod
    def _markdown(sources: list[dict[str, Any]], rows: list[dict[str, Any]], created: str) -> str:
        sections = ["# 模型性能汇总", f"生成时间（UTC）：{created}。共 {len(sources)} 份输入，{len(rows)} 个分组汇总。",
                    "直接提取原报告统计，不合并运行、不重新评估。all 取原始整体统计，不平均数据集均值。",
                    "## 来源与评估协议"]
        sections.append(_table(
            ["来源", "报告路径", "ZIP", "数据集", "Fingerprint", "Content digest", "Split", "来源划分", "协议"],
            [[source[key] for key in ("source_id", "source_report", "archive_filename", "dataset_reference",
                                     "dataset_fingerprint", "dataset_content_digest", "dataset_split",
                                     "dataset_source_assignment", "protocol_id")] for source in sources],
        ))
        protocols = {source["protocol_id"]: source["evaluation_protocol"] for source in sources}
        for name, encoded in protocols.items():
            protocol = json.loads(encoded)
            sections.append(f"### {name}")
            sections.append("\n".join(
                f"- {_cell(key)}：{_cell(_json(value) if isinstance(value, dict) else value)}"
                for key, value in protocol.items()
            ))
        sections.extend([
            "## 实验设置",
            _table(["来源", "参数", "方法", "生成模型", "Revision", "分辨率", "步数", "Seed", "编辑 CFG",
                    "Cross", "Self", "方法参数"], [
                [row["source_id"], row["parameter_file"], row["method"], row["model_id"], row["revision"],
                 f"{row['width']}×{row['height']}", row["num_inference_steps"], row["seed"],
                 row["editing_guidance_scale"], row["cross_replace_fraction"], row["self_replace_fraction"],
                 row["method_h_params"]] for row in rows if row["group"] == "all"
            ]),
            "## 分组统计",
            "质量单元格为均值 ± 样本标准差（n=有效样本数）。— 表示缺失或未定义，+∞ 表示无限 PSNR。"
            "Markdown 数值显示四位小数；CSV 保留原始精度。CLIP 为原始余弦相似度。",
            "耗时单位为秒，是批次均摊的观测核心时间，并非单图实测延迟。缓存命中可使反演耗时为零，"
            "不代表原始反演成本；未知时间不当作零。耗时不含模型加载、外部 I/O、准备与发布，"
            "编辑包含源图重建和目标编辑。计时 n 与缓存已知数单独列出，评估耗时不作为实验耗时。",
        ])
        for group in GROUPS:
            selected = [row for row in rows if row["group"] == group]
            sections.extend([f"### {group}", "质量：", _table(
                ["来源", "参数", "方法", "样本数", "CLIP ↑", "区域 CLIP ↑", "背景 PSNR (dB) ↑",
                 "背景 SSIM ↑", "背景 LPIPS ↓"], [
                    [row["source_id"], row["parameter_file"], row["method"], row["sample_count"], *(
                        f"{_cell(row[metric + '_mean'])} ± {_cell(row[metric + '_std'])} "
                        f"(n={row[metric + '_n']})" for metric in METRICS
                    )] for row in selected
                ]), "耗时与缓存：", _table(
                    ["来源", "参数", "方法", "反演 n", "反演总秒", "反演秒/图", "编辑 n", "编辑总秒",
                     "编辑秒/图", "缓存已知数", "缓存命中数", "缓存命中率"], [
                        [row[key] for key in ("source_id", "parameter_file", "method", "inversion_n",
                         "inversion_total_core_seconds", "inversion_mean_core_seconds_per_image", "editing_n",
                         "editing_total_core_seconds", "editing_mean_core_seconds_per_image",
                         "inversion_cache_known_count", "inversion_cache_hit_count", "inversion_cache_hit_rate")]
                        for row in selected
                    ]),
            ])
        return "\n\n".join(sections) + "\n"

    def run(self) -> Path:
        """Validate all inputs before atomically publishing a fresh output directory."""
        sources, rows, protocols = [], [], {}
        for index, path in enumerate(self.reports, start=1):
            try:
                with path.open(encoding="utf-8") as stream:
                    document = json.load(stream)
                source, extracted = self._extract(_object(document, "report"), path, f"R{index}")
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid metrics report {path}: {exc}") from exc
            protocol_id = protocols.setdefault(source["evaluation_protocol"], f"P{len(protocols) + 1}")
            source["protocol_id"] = protocol_id
            for row in extracted:
                row["protocol_id"] = protocol_id
            sources.append(source)
            rows.extend(extracted)
        now = datetime.now(UTC)
        markdown = self._markdown(sources, rows, now.isoformat())
        suffix = uuid4().hex[:8]
        destination = self.output_root / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + suffix)
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging = self.output_root / (".tmp-" + suffix)
        staging.mkdir()
        try:
            (staging / "report.md").write_text(markdown, encoding="utf-8")
            # A BOM lets Excel recognize UTF-8 source paths and method settings.
            with (staging / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            staging.rename(destination)
        except BaseException as exc:
            try:
                for name in ("report.md", "summary.csv"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
            except OSError as cleanup:
                exc.add_note(f"Could not remove temporary summary directory: {cleanup}")
            raise
        return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize existing metrics JSON as Markdown and CSV.")
    parser.add_argument("reports", nargs="*", type=Path, metavar="REPORT_OR_DIR",
                        help="JSON files or directories relative to cwd; omitted selects data/metrics recursively.")
    args = parser.parse_args(argv)
    try:
        destination = MetricsSummarizer(args.reports or None).run()
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
