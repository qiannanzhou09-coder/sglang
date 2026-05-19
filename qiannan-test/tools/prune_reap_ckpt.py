#!/usr/bin/env python3
"""
prune_reap_ckpt.py
==================

Apply a REAP MoE pruning plan to a ``Qwen/Qwen3.5-122B-A10B-FP8``-style
checkpoint at the safetensors-file level, and/or strip the ViT vision tower
for a text-only variant. No forward pass, no calibration.

The Qwen3.5-122B-A10B-FP8 release ships **three** weight families in
``model.safetensors.index.json``:

1. ``model.language_model.*`` — 48-layer MoE LM (each layer has 256 routed
   experts + 1 shared expert + linear/self attention + norms).
2. ``model.visual.*``         — 27-layer ViT vision tower + merger +
   patch_embed + pos_embed.
3. ``mtp.*``                  — 1-layer Multi-Token-Prediction (NEXTN) head
   with its own 256 routed experts, shared expert, gate, self-attn and
   layernorms, plus top-level ``mtp.{fc,norm,pre_fc_norm_*}`` projections.

The supported modes (any combination is OK):

A. ``--strip-visual`` (alone): produces a vision-stripped baseline by
   dropping ``model.visual.*`` and removing ``vision_config`` and the
   image/video token id fields from ``config.json``. Useful as a cleaner
   baseline for text-only KV-cache experiments.

B. ``--plan PATH``: applies the REAP pruning plan to the LM and, by
   default, mirrors the layer-0 plan onto the MTP head (see
   ``--mtp-policy`` below). Required when shrinking ``num_experts``.

C. ``--plan PATH --strip-visual``: combines (A) and (B).

What the LM pruning does, per layer in the plan:

1. Removes the FFN weights of the pruned routed experts
   (``model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,weight_scale_inv}``).
2. Re-indexes the surviving experts to ``0..K-1``
   (where ``K = num_experts - pruned_per_layer``).
3. Shrinks the router ``...mlp.gate.weight`` from ``[num_experts, H]`` to
   ``[K, H]`` by selecting the rows of the surviving experts.
4. Leaves ``shared_expert.*``, ``shared_expert_gate.weight``,
   ``linear_attn.*``, ``self_attn.*``, all norms, ``lm_head`` and
   ``embed_tokens`` untouched.
5. Patches ``text_config.num_experts`` in ``config.json`` from
   ``num_experts`` to ``K``.

``--mtp-policy`` (only meaningful in modes B/C, where ``num_experts``
shrinks and any 256-expert MTP head would no longer match the new
config):

* ``prune-with-lm0`` (default) — apply the LM layer-0 plan to the MTP
  head's single layer. This is what keeps ``num_experts`` consistent
  across the whole ckpt without needing a separate REAP run for MTP.
  Defensible because the MTP head consumes the LM's final hidden state,
  so its first-layer routing correlates with LM layer 0.
* ``drop`` — remove every ``mtp.*`` tensor and set
  ``text_config.mtp_num_hidden_layers = 0``. Disables NEXTN speculative
  decoding.
* ``keep`` — leave MTP untouched. *Refused when --plan is set* because it
  would produce a checkpoint where ``config.num_experts`` disagrees with
  the MTP router shape (256 rows) and gate, breaking model load.

Notes on quantization layout
----------------------------
* FP8 storage uses two tensors per quantized linear: ``X.weight``
  (``float8_e4m3fn``) and ``X.weight_scale_inv`` (``float32`` per-block
  scale, block size 128x128). Both are pruned together via the shared
  ``mlp.experts.{E}.<...>`` prefix, yielding 6 tensors per expert
  (``{gate,up,down}_proj`` x ``{weight, weight_scale_inv}``).
* The router (``mlp.gate.weight``) and the shared-expert gate
  (``mlp.shared_expert_gate.weight``) are **not** quantized; only
  ``.weight`` exists. We only need to row-shrink the router.

Streaming design
----------------
One safetensors shard at a time; peak RAM ~= size of largest shard
(~3 GB for the FP8 release x 39 shards = ~127 GB on disk). CPU only.

Usage
-----

::

    # REAP-20-FP8 (keep vision; prune LM + MTP)
    python tools/prune_reap_ckpt.py \\
        --src   /home/Qwen3.5-122B-A10B \\
        --plan  qiannan-test/document/targeted_refusal_analysis.json \\
        --dst   /home/qwen3.5-pruned-models/Qwen3.5-122B-A10B-REAP-20

    # REAP-20-FP8-text-only (prune LM + MTP, drop ViT)
    python tools/prune_reap_ckpt.py \\
        --src   /home/Qwen3.5-122B-A10B \\
        --plan  qiannan-test/document/targeted_refusal_analysis.json \\
        --dst   /home/qwen3.5-pruned-models/Qwen3.5-122B-A10B-REAP-20-text-only \\
        --strip-visual

    # baseline-FP8-text-only (no pruning, just drop ViT)
    python tools/prune_reap_ckpt.py \\
        --src   /home/Qwen3.5-122B-A10B \\
        --dst   /home/qwen3.5-pruned-models/Qwen3.5-122B-A10B-text-only \\
        --strip-visual

After the script finishes, ``--dst`` is a self-contained model directory
that can be loaded by SGLang / vLLM / Transformers directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prune_reap")

# --------------------------------------------------------------------------- #
# Tensor name prefixes / matchers (strictly anchored — no .*? leakage)
# --------------------------------------------------------------------------- #
LM_PREFIX = "model.language_model.layers."
MTP_PREFIX = "mtp.layers."
VISUAL_PREFIX = "model.visual."

LM_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)(?P<suffix>\..+)$"
)
LM_GATE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$"
)

MTP_EXPERT_RE = re.compile(
    r"^mtp\.layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)(?P<suffix>\..+)$"
)
MTP_GATE_RE = re.compile(
    r"^mtp\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$"
)

# Number of tensors that one routed expert contributes (gate/up/down × weight/scale_inv)
TENSORS_PER_EXPERT = 6

# --------------------------------------------------------------------------- #
# Plan parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerPlan:
    layer: int
    pruned: tuple[int, ...]  # sorted ascending, len == num_pruned
    kept: tuple[int, ...]  # sorted ascending, len == num_experts - num_pruned
    old_to_new: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def build(layer: int, pruned: list[int], num_experts: int) -> "LayerPlan":
        pruned_sorted = tuple(sorted(set(int(x) for x in pruned)))
        pruned_set = set(pruned_sorted)
        if any(i < 0 or i >= num_experts for i in pruned_sorted):
            raise ValueError(
                f"layer {layer}: pruned id out of range [0,{num_experts}): {pruned_sorted}"
            )
        kept = tuple(i for i in range(num_experts) if i not in pruned_set)
        remap = {old: new for new, old in enumerate(kept)}
        return LayerPlan(layer=layer, pruned=pruned_sorted, kept=kept, old_to_new=remap)


def load_plan(path: Path, num_experts: int) -> dict[int, LayerPlan]:
    data = json.loads(path.read_text())
    if "layers" not in data:
        raise ValueError(f"{path}: missing top-level 'layers' key")
    plans: dict[int, LayerPlan] = {}
    counts = set()
    for entry in data["layers"]:
        layer = int(entry["layer"])
        pruned = entry["pruned_indices"]
        plans[layer] = LayerPlan.build(layer, pruned, num_experts)
        counts.add(len(plans[layer].pruned))
    if len(counts) != 1:
        log.warning(
            "Plan has non-uniform pruning per layer (sizes=%s); the resulting "
            "ckpt cannot be loaded by a single num_experts config — refusing.",
            sorted(counts),
        )
        raise SystemExit(2)
    if 0 not in plans:
        raise SystemExit(
            "Plan is missing layer 0; it's required to mirror onto the MTP head."
        )
    n_pruned = next(iter(counts))
    new_num_experts = num_experts - n_pruned
    log.info(
        "Plan parsed: %d LM layers, %d experts pruned/layer (%d → %d kept)",
        len(plans),
        n_pruned,
        num_experts,
        new_num_experts,
    )
    return plans


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def read_config(src: Path) -> dict[str, Any]:
    return json.loads((src / "config.json").read_text())


def get_num_experts(config: dict[str, Any]) -> int:
    """Find ``num_experts`` in the config (Qwen3.5 puts it under ``text_config``)."""
    candidates = []
    for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
        if key in config:
            candidates.append((key, config[key], None))
    for sub_key in ("text_config", "language_model_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
                if key in config[sub_key]:
                    candidates.append((key, config[sub_key][key], sub_key))
    if not candidates:
        raise RuntimeError("Could not find num_experts in config.json")
    vals = {(c[1]) for c in candidates}
    if len(vals) != 1:
        raise RuntimeError(f"num_experts mismatch across config keys: {candidates}")
    return int(next(iter(vals)))


def set_num_experts(config: dict[str, Any], new_num: int) -> int:
    """Patch every occurrence; return how many keys were patched."""
    patched = 0
    for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
        if key in config:
            config[key] = new_num
            patched += 1
    for sub_key in ("text_config", "language_model_config"):
        if sub_key in config and isinstance(config[sub_key], dict):
            for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
                if key in config[sub_key]:
                    config[sub_key][key] = new_num
                    patched += 1
    return patched


VISION_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def patch_config(
    config: dict[str, Any],
    *,
    new_num_experts: int | None,
    drop_mtp: bool,
    strip_visual: bool,
) -> dict[str, Any]:
    """Return a deep-copied + patched config.json dict.

    * Patches ``num_experts`` everywhere if ``new_num_experts`` given.
    * Sets ``mtp_num_hidden_layers`` to 0 if ``drop_mtp``.
    * Drops every vision-related field at top level and under ``text_config``
      if ``strip_visual``. ``architectures`` is intentionally untouched
      (downstream code may still instantiate the multimodal class but skip
      the vision branch when ``vision_config`` is absent).
    * Prunes ``quantization_config.modules_to_not_convert`` entries that
      reference removed modules.
    """
    cfg = json.loads(json.dumps(config))  # deep copy

    if new_num_experts is not None:
        n_patched = set_num_experts(cfg, new_num_experts)
        log.info(
            "Patched num_experts in %d config key(s) → %d",
            n_patched,
            new_num_experts,
        )

    if drop_mtp:
        n_mtp = 0
        for parent in (cfg, cfg.get("text_config"), cfg.get("language_model_config")):
            if isinstance(parent, dict) and "mtp_num_hidden_layers" in parent:
                parent["mtp_num_hidden_layers"] = 0
                n_mtp += 1
        log.info("Disabled MTP: set mtp_num_hidden_layers=0 in %d config key(s)", n_mtp)

    if strip_visual:
        n_visual = 0
        for parent in (cfg, cfg.get("text_config")):
            if not isinstance(parent, dict):
                continue
            for key in VISION_CONFIG_KEYS:
                if key in parent:
                    parent.pop(key)
                    n_visual += 1
        log.info("Stripped %d vision-related field(s) from config", n_visual)

    qcfg = cfg.get("quantization_config")
    if isinstance(qcfg, dict) and isinstance(qcfg.get("modules_to_not_convert"), list):
        before = len(qcfg["modules_to_not_convert"])
        kept = []
        for mod in qcfg["modules_to_not_convert"]:
            if drop_mtp and mod.startswith("mtp."):
                continue
            if strip_visual and mod.startswith("model.visual."):
                continue
            kept.append(mod)
        qcfg["modules_to_not_convert"] = kept
        if len(kept) != before:
            log.info(
                "Pruned modules_to_not_convert: %d → %d entries", before, len(kept)
            )

    return cfg


# --------------------------------------------------------------------------- #
# Shard discovery
# --------------------------------------------------------------------------- #
def collect_shards(src: Path) -> tuple[list[Path], dict[str, str] | None]:
    """
    Return (sorted list of shard files, original weight_map or None).

    Prefers ``model.safetensors.index.json``; falls back to glob.
    """
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        weight_map = idx["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        return [src / s for s in shard_names], weight_map

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors found under {src}")
    return shards, None


# --------------------------------------------------------------------------- #
# Per-shard processing
# --------------------------------------------------------------------------- #
@dataclass
class ShardStats:
    shard_name: str
    out_shard_name: str | None
    kept: int = 0
    renamed: int = 0
    gate_shrunk_lm: int = 0
    gate_shrunk_mtp: int = 0
    dropped_expert_lm: int = 0
    dropped_expert_mtp: int = 0
    dropped_visual: int = 0
    dropped_mtp_other: int = 0  # MTP non-expert tensors when mtp_action == "drop"
    out_bytes: int = 0
    new_names: list[str] = field(default_factory=list)


def _classify(
    name: str,
    plans: dict[int, LayerPlan],
    mtp_action: str,  # "prune-with-lm0" | "keep" | "drop"
    strip_visual: bool,
) -> tuple[str, str | None, int | None, str]:
    """
    Decide what to do with one tensor name.

    Returns ``(action, new_name, layer, tag)`` where:
      * ``action`` ∈ {"keep", "rename", "drop", "gate_shrink"}
      * ``new_name`` set only for ``rename``
      * ``layer`` is the plan-layer to use for ``gate_shrink``
      * ``tag`` is a fine-grained stats bucket (e.g. ``"drop_expert_lm"``,
        ``"drop_visual"``, ``"gate_shrink_mtp"``, ...) — used purely for
        diagnostics; the action is what process_shard branches on.
    """
    # ---- Visual encoder ---------------------------------------------------
    if name.startswith(VISUAL_PREFIX):
        if strip_visual:
            return "drop", None, None, "drop_visual"
        return "keep", None, None, "keep"

    # ---- MTP head ---------------------------------------------------------
    if name.startswith("mtp."):
        if mtp_action == "drop":
            # Distinguish expert tensors (predictable count) from the rest
            if MTP_EXPERT_RE.match(name):
                return "drop", None, None, "drop_expert_mtp"
            return "drop", None, None, "drop_mtp_other"
        if mtp_action == "keep":
            return "keep", None, None, "keep"

        # mtp_action == "prune-with-lm0": mirror plans[0] onto MTP layer 0
        e = MTP_EXPERT_RE.match(name)
        if e:
            plan = plans[0]
            old_expert = int(e.group("expert"))
            if old_expert in set(plan.pruned):
                return "drop", None, None, "drop_expert_mtp"
            new_expert = plan.old_to_new[old_expert]
            if new_expert == old_expert:
                return "keep", None, None, "keep"
            new_name = (
                f"mtp.layers.{e.group('layer')}.mlp.experts.{new_expert}"
                f"{e.group('suffix')}"
            )
            return "rename", new_name, None, "rename"

        g = MTP_GATE_RE.match(name)
        if g:
            return "gate_shrink", None, 0, "gate_shrink_mtp"

        # Other mtp.* (norms, shared_expert, self_attn, fc, pre_fc_norm_*) → keep
        return "keep", None, None, "keep"

    # ---- Main model LM layers --------------------------------------------
    g = LM_GATE_RE.match(name)
    if g:
        layer = int(g.group("layer"))
        if layer in plans:
            return "gate_shrink", None, layer, "gate_shrink_lm"
        return "keep", None, None, "keep"

    e = LM_EXPERT_RE.match(name)
    if e:
        layer = int(e.group("layer"))
        if layer not in plans:
            return "keep", None, None, "keep"
        plan = plans[layer]
        old_expert = int(e.group("expert"))
        if old_expert in set(plan.pruned):
            return "drop", None, None, "drop_expert_lm"
        new_expert = plan.old_to_new[old_expert]
        if new_expert == old_expert:
            return "keep", None, None, "keep"
        new_name = (
            f"model.language_model.layers.{e.group('layer')}"
            f".mlp.experts.{new_expert}{e.group('suffix')}"
        )
        return "rename", new_name, None, "rename"

    # ---- Everything else (lm_head, embed_tokens, shared_expert, ...) -----
    return "keep", None, None, "keep"


def process_shard(
    src_shard: Path,
    dst_dir: Path,
    plans_pickle: dict[int, dict[str, Any]],
    expected_num_experts: int,
    mtp_action: str,
    strip_visual: bool,
    dry_run: bool,
) -> ShardStats:
    """Read one shard, write its pruned counterpart. Designed to be subprocess-safe."""
    plans = {
        layer: LayerPlan(
            layer=p["layer"],
            pruned=tuple(p["pruned"]),
            kept=tuple(p["kept"]),
            old_to_new={int(k): v for k, v in p["old_to_new"].items()},
        )
        for layer, p in plans_pickle.items()
    }

    stats = ShardStats(shard_name=src_shard.name, out_shard_name=None)
    out_tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    with safe_open(str(src_shard), framework="pt") as f:
        keys = list(f.keys())
        for name in keys:
            action, new_name, layer, tag = _classify(
                name, plans, mtp_action, strip_visual
            )

            if action == "drop":
                if tag == "drop_expert_lm":
                    stats.dropped_expert_lm += 1
                elif tag == "drop_expert_mtp":
                    stats.dropped_expert_mtp += 1
                elif tag == "drop_visual":
                    stats.dropped_visual += 1
                elif tag == "drop_mtp_other":
                    stats.dropped_mtp_other += 1
                continue

            if action == "gate_shrink":
                plan = plans[layer]
                tensor = f.get_tensor(name)
                if tensor.dim() < 1 or tensor.shape[0] != expected_num_experts:
                    raise RuntimeError(
                        f"{src_shard.name}: gate tensor {name} has shape "
                        f"{tuple(tensor.shape)}, expected first dim == "
                        f"{expected_num_experts}"
                    )
                kept_idx = torch.as_tensor(list(plan.kept), dtype=torch.long)
                new_tensor = tensor.index_select(0, kept_idx).contiguous()
                out_tensors[name] = new_tensor
                if tag == "gate_shrink_lm":
                    stats.gate_shrunk_lm += 1
                else:
                    stats.gate_shrunk_mtp += 1
                continue

            # 'keep' / 'rename' — both copy the tensor data verbatim
            tensor = f.get_tensor(name)
            target_name = new_name if action == "rename" else name
            out_tensors[target_name] = tensor
            if action == "rename":
                stats.renamed += 1
            else:
                stats.kept += 1

        if dry_run:
            return stats

        if not out_tensors:
            # Shard became entirely empty (very unlikely). Skip writing.
            log.warning("%s: all tensors dropped, no output shard written", src_shard.name)
            return stats

        out_path = dst_dir / src_shard.name
        save_file(
            {k: v for k, v in out_tensors.items()},
            str(out_path),
            metadata={"format": "pt"},
        )
        stats.out_shard_name = src_shard.name
        stats.out_bytes = out_path.stat().st_size
        stats.new_names = list(out_tensors.keys())

    return stats


# --------------------------------------------------------------------------- #
# Companion files (tokenizer / chat template / preprocessor / ...)
# --------------------------------------------------------------------------- #
COMPANION_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
)


def copy_companion_files(src: Path, dst: Path, *, strip_visual: bool) -> int:
    """Copy tokenizer / chat template / preprocessor files.

    When ``strip_visual``, the image/video preprocessor configs are still
    copied verbatim (the model loader may ignore them when ``vision_config``
    is absent; removing them risks breaking unrelated tooling that touches
    AutoProcessor).
    """
    n = 0
    for fname in COMPANION_FILES:
        p = src / fname
        if p.exists():
            shutil.copy2(p, dst / fname)
            n += 1
    return n


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, type=Path, help="Source model directory")
    ap.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="REAP pruning plan JSON. Omit to skip expert pruning (e.g. "
        "to make a vision-stripped baseline together with --strip-visual).",
    )
    ap.add_argument("--dst", required=True, type=Path, help="Output model directory")
    ap.add_argument(
        "--strip-visual",
        action="store_true",
        help="Drop all model.visual.* tensors and remove vision_config / "
        "image_token_id / video_token_id / vision_*_token_id fields from "
        "config.json. The 'architectures' field is intentionally NOT changed "
        "(the multimodal class is expected to lazily skip the vision branch "
        "when vision_config is absent).",
    )
    ap.add_argument(
        "--mtp-policy",
        choices=("prune-with-lm0", "keep", "drop"),
        default="prune-with-lm0",
        help="What to do with the 1-layer MTP head when --plan shrinks "
        "num_experts. 'prune-with-lm0' (default): mirror the LM layer-0 "
        "pruning onto the MTP layer so num_experts stays consistent. "
        "'drop': remove all mtp.* tensors and disable MTP in config. "
        "'keep': leave MTP untouched (refused when --plan is set, since "
        "the resulting num_experts mismatch would break model load).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default 1; 2–4 helps "
        "overlap disk I/O if you have spare RAM)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan shards, print counts; do not write any files",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --dst if it already exists",
    )
    args = ap.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()
    plan_path: Path | None = args.plan.expanduser().resolve() if args.plan else None

    # ---- Argument validation ---------------------------------------------
    if not src.is_dir():
        raise SystemExit(f"--src not a directory: {src}")
    if plan_path is not None and not plan_path.is_file():
        raise SystemExit(f"--plan not a file: {plan_path}")
    if plan_path is None and not args.strip_visual:
        raise SystemExit(
            "Nothing to do: pass --plan and/or --strip-visual; otherwise the "
            "output would be identical to --src."
        )
    if plan_path is not None and args.mtp_policy == "keep":
        raise SystemExit(
            "--plan shrinks num_experts but --mtp-policy keep leaves MTP at the "
            "original expert count → ckpt would fail to load. Use prune-with-lm0 "
            "(default) or drop."
        )
    if dst.exists():
        if args.force and not args.dry_run:
            log.info("Removing existing %s (--force)", dst)
            shutil.rmtree(dst)
        elif not args.dry_run:
            raise SystemExit(
                f"--dst already exists: {dst}\n  use --force to overwrite, or pick a new path"
            )

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=False)

    # ---- Derive effective mtp_action -------------------------------------
    # When --plan is absent, "prune-with-lm0" is meaningless (no plan to mirror).
    # Coerce it to "keep" so the user can pass --mtp-policy drop standalone to
    # produce e.g. a vision-stripped + MTP-stripped baseline.
    if plan_path is None:
        mtp_action = "drop" if args.mtp_policy == "drop" else "keep"
        if args.mtp_policy == "prune-with-lm0":
            log.info(
                "--plan absent: --mtp-policy=prune-with-lm0 has no effect, "
                "MTP will be kept verbatim"
            )
    else:
        mtp_action = args.mtp_policy  # already validated not to be "keep"

    # ---- Load config & detect num_experts --------------------------------
    config = read_config(src)
    num_experts_old = get_num_experts(config)
    log.info(
        "Source: %s  (num_experts=%d, strip_visual=%s, mtp_action=%s)",
        src,
        num_experts_old,
        args.strip_visual,
        mtp_action,
    )

    # ---- Parse plan with the detected num_experts ------------------------
    if plan_path is not None:
        plans = load_plan(plan_path, num_experts_old)
        n_pruned_per_layer = len(next(iter(plans.values())).pruned)
        num_experts_new = num_experts_old - n_pruned_per_layer
    else:
        plans = {}
        n_pruned_per_layer = 0
        num_experts_new = num_experts_old

    # ---- Discover shards -------------------------------------------------
    shards, _orig_weight_map = collect_shards(src)
    log.info(
        "Found %d safetensors shard(s) totalling %.2f GB",
        len(shards),
        sum(s.stat().st_size for s in shards) / 1e9,
    )

    # ---- Picklable plans for subprocesses --------------------------------
    plans_pickle = {
        layer: {
            "layer": p.layer,
            "pruned": list(p.pruned),
            "kept": list(p.kept),
            "old_to_new": p.old_to_new,
        }
        for layer, p in plans.items()
    }

    # ---- Process every shard --------------------------------------------
    new_weight_map: dict[str, str] = {}
    totals = defaultdict(int)
    total_out_bytes = 0

    if args.workers <= 1:
        for shard_idx, shard in enumerate(shards, start=1):
            log.info("[%d/%d] %s", shard_idx, len(shards), shard.name)
            stats = process_shard(
                shard,
                dst,
                plans_pickle,
                num_experts_old,
                mtp_action,
                args.strip_visual,
                args.dry_run,
            )
            _accum_stats(stats, totals, new_weight_map)
            total_out_bytes += stats.out_bytes
    else:
        log.info("Spawning %d worker process(es)", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_shard,
                    shard,
                    dst,
                    plans_pickle,
                    num_experts_old,
                    mtp_action,
                    args.strip_visual,
                    args.dry_run,
                ): shard
                for shard in shards
            }
            done_idx = 0
            for fut in as_completed(futures):
                stats = fut.result()
                done_idx += 1
                log.info("[%d/%d] %s done", done_idx, len(shards), stats.shard_name)
                _accum_stats(stats, totals, new_weight_map)
                total_out_bytes += stats.out_bytes

    # ---- Expected counts (precise where we can) --------------------------
    expected_gate_lm = len(plans)
    expected_gate_mtp = 1 if (plans and mtp_action == "prune-with-lm0") else 0
    expected_drop_lm = len(plans) * n_pruned_per_layer * TENSORS_PER_EXPERT
    expected_drop_mtp = (
        n_pruned_per_layer * TENSORS_PER_EXPERT
        if (plans and mtp_action == "prune-with-lm0")
        else 0
    )

    log.info("=== Tensor count summary ===")
    log.info("  kept (verbatim)        : %d", totals["kept"])
    log.info("  renamed (expert id)    : %d", totals["renamed"])
    log.info(
        "  LM router shrunk       : %d  (expected %d)",
        totals["gate_shrunk_lm"],
        expected_gate_lm,
    )
    log.info(
        "  MTP router shrunk      : %d  (expected %d)",
        totals["gate_shrunk_mtp"],
        expected_gate_mtp,
    )
    log.info(
        "  LM experts dropped     : %d  (expected %d)",
        totals["dropped_expert_lm"],
        expected_drop_lm,
    )
    log.info(
        "  MTP experts dropped    : %d  (expected %d)",
        totals["dropped_expert_mtp"],
        expected_drop_mtp,
    )
    log.info(
        "  visual tensors dropped : %d%s",
        totals["dropped_visual"],
        "" if args.strip_visual else "  (--strip-visual not set, expected 0)",
    )
    log.info(
        "  other MTP tensors drop : %d%s",
        totals["dropped_mtp_other"],
        "" if mtp_action == "drop" else "  (mtp_action != drop, expected 0)",
    )
    log.info("  output bytes           : %.2f GB", total_out_bytes / 1e9)

    # ---- Hard assertions on the precisely-known counts -------------------
    errors = []
    if totals["gate_shrunk_lm"] != expected_gate_lm:
        errors.append(
            f"LM router shrink mismatch: got {totals['gate_shrunk_lm']}, "
            f"expected {expected_gate_lm}"
        )
    if totals["gate_shrunk_mtp"] != expected_gate_mtp:
        errors.append(
            f"MTP router shrink mismatch: got {totals['gate_shrunk_mtp']}, "
            f"expected {expected_gate_mtp}"
        )
    if totals["dropped_expert_lm"] != expected_drop_lm:
        errors.append(
            f"LM expert drop mismatch: got {totals['dropped_expert_lm']}, "
            f"expected {expected_drop_lm}"
        )
    if totals["dropped_expert_mtp"] != expected_drop_mtp:
        errors.append(
            f"MTP expert drop mismatch: got {totals['dropped_expert_mtp']}, "
            f"expected {expected_drop_mtp}"
        )
    if errors:
        for err in errors:
            log.error(err)
        log.error(
            "Aborting before writing index/config. Inspect/delete %s manually.",
            dst,
        )
        raise SystemExit(3)

    if args.dry_run:
        log.info("--dry-run: no files written")
        return

    # ---- Write new index -------------------------------------------------
    write_new_index(dst, new_weight_map, total_out_bytes)

    # ---- Patch config.json and copy companion files ----------------------
    new_config = patch_config(
        config,
        new_num_experts=num_experts_new if plans else None,
        drop_mtp=(mtp_action == "drop"),
        strip_visual=args.strip_visual,
    )
    (dst / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")

    n_copied = copy_companion_files(src, dst, strip_visual=args.strip_visual)
    log.info("Copied %d companion file(s)", n_copied)

    log.info("✓ Done. Output: %s", dst)


def _accum_stats(
    stats: ShardStats,
    totals: dict[str, int],
    weight_map: dict[str, str],
) -> None:
    totals["kept"] += stats.kept
    totals["renamed"] += stats.renamed
    totals["gate_shrunk_lm"] += stats.gate_shrunk_lm
    totals["gate_shrunk_mtp"] += stats.gate_shrunk_mtp
    totals["dropped_expert_lm"] += stats.dropped_expert_lm
    totals["dropped_expert_mtp"] += stats.dropped_expert_mtp
    totals["dropped_visual"] += stats.dropped_visual
    totals["dropped_mtp_other"] += stats.dropped_mtp_other
    if stats.out_shard_name:
        for tn in stats.new_names:
            weight_map[tn] = stats.out_shard_name


def write_new_index(dst: Path, weight_map: dict[str, str], total_size: int) -> None:
    # NOTE: ``total_size`` here is the on-disk byte sum (includes safetensors
    # header overhead). HF convention is ``sum(numel × itemsize)``; this is an
    # over-approximation that loaders use only for display. SGLang / vLLM do
    # not depend on it being exact.
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    out = dst / "model.safetensors.index.json"
    out.write_text(json.dumps(index, indent=2) + "\n")
    log.info("Wrote %s (%d entries)", out.name, len(weight_map))


if __name__ == "__main__":
    main()
