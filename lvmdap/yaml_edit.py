#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yaml_edit.py
============

Programmatic version of the ``_yaml_edit`` CLI: edit a YAML file by
applying a dictionary of overrides, instead of parsing ``sys.argv``.

    from yaml_edit import edit_yaml

    edit_yaml("config.yaml", "config_new.yaml", {
        "nside": 128,
        "gaia.release": "dr3",
        "chunks[0].level": 1,
        "filters.rv": [0.0, 5.0],
    })

Keys are dotted paths that may contain list indices, exactly as in the
command-line version: ``a.b[0].c`` reaches ``data["a"]["b"][0]["c"]``.
Values are used as given, so they can be any Python object that PyYAML
can serialise; set ``parse_values=True`` if they arrive as strings from
some other CLI-ish source and you want ``"3"`` to become ``3``.

By default a missing key is an error, which is usually what you want when
overriding a config: a typo in a key name fails loudly instead of quietly
adding a setting nothing reads.  Pass ``create_missing=True`` to build
intermediate mappings along the path.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

__all__ = ["edit_yaml", "set_by_path", "split_path", "parse_value"]


_PATH_TOKEN_RE = re.compile(
    r"""
    (?P<name>[^.\[\]]+)     # dict key
    (?P<idx>(\[[0-9]+\])*)  # optional [0][1]...
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _SetOptions:
    create_missing: bool = False


def parse_value(raw: str) -> Any:
    """Parse a string into Python types (YAML first, then literal for tuples)."""
    try:
        val = yaml.safe_load(raw)
    except Exception:
        val = raw

    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                val2 = ast.literal_eval(s)
                val = val2
            except Exception:
                pass

    if isinstance(val, tuple):
        val = list(val)

    return val


def split_path(key_path: str) -> List[Any]:
    """Split dotted path with optional list indices: a.b[0].c -> ['a','b',0,'c']."""
    parts: List[Any] = []
    for seg in key_path.split("."):
        m = _PATH_TOKEN_RE.fullmatch(seg)
        if not m:
            raise ValueError(f"Invalid key path segment: '{seg}' in '{key_path}'")
        parts.append(m.group("name"))
        idx_part = m.group("idx") or ""
        if idx_part:
            for idx_str in re.findall(r"\[([0-9]+)\]", idx_part):
                parts.append(int(idx_str))
    return parts


def set_by_path(obj: Any, key_path: str, value: Any, opts: _SetOptions) -> None:
    """Set obj[key_path] = value, where key_path may contain dots and [idx]."""
    tokens = split_path(key_path)
    cur = obj
    for t in tokens[:-1]:
        if isinstance(t, int):
            if not isinstance(cur, list):
                raise KeyError(f"Path '{key_path}': expected list, got {type(cur).__name__}")
            if t < 0 or t >= len(cur):
                raise IndexError(f"Path '{key_path}': list index {t} out of range (len={len(cur)})")
            cur = cur[t]
        else:
            if not isinstance(cur, dict):
                raise KeyError(f"Path '{key_path}': expected dict, got {type(cur).__name__}")
            if t not in cur:
                if not opts.create_missing:
                    raise KeyError(f"Key '{t}' not found while setting '{key_path}'.")
                cur[t] = {}
            cur = cur[t]

    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(cur, list):
            raise KeyError(f"Path '{key_path}': expected list at final parent, got {type(cur).__name__}")
        if last < 0 or last >= len(cur):
            raise IndexError(f"Path '{key_path}': list index {last} out of range (len={len(cur)})")
        cur[last] = value
        return

    if not isinstance(cur, dict):
        raise KeyError(f"Path '{key_path}': expected dict at final parent, got {type(cur).__name__}")
    if (last not in cur) and (not opts.create_missing):
        raise KeyError(f"Key '{last}' not found while setting '{key_path}'.")
    cur[last] = value


def edit_yaml(
    input_path: str,
    output_path: Optional[str],
    overrides: Mapping[str, Any],
    *,
    create_missing: bool = False,
    parse_values: bool = False,
) -> Dict[str, Any]:
    """Read a YAML file, apply ``overrides``, and write the result.

    Parameters
    ----------
    input_path : str
        YAML file to read.  Its top level must be a mapping.
    output_path : str or None
        Where to write the edited YAML.  Pass the same string as
        ``input_path`` to edit in place, or ``None`` to skip writing and
        only get the modified structure back.
    overrides : Mapping[str, Any]
        Flat mapping of dotted path -> new value, e.g.
        ``{"a.b[0].c": 1}``.  It is flat, not nested: a dict value is
        assigned whole, so ``{"gaia": {"release": "dr3"}}`` *replaces*
        the entire ``gaia`` node, while ``{"gaia.release": "dr3"}``
        changes only that one field.
    create_missing : bool
        Create intermediate mappings and new leaf keys instead of raising
        ``KeyError``.  Note this cannot create list elements: an index
        must already exist.
    parse_values : bool
        Interpret string values with :func:`parse_value` (YAML scalar
        rules, plus tuple/list/dict literals).  Leave ``False`` when the
        values are already proper Python objects, otherwise a string like
        ``"1.0"`` silently becomes a float.

    Returns
    -------
    dict
        The modified structure, whether or not it was written.

    Notes
    -----
    Round-tripping through ``yaml.safe_load``/``safe_dump`` **discards
    comments, anchors and original formatting**.  Key order is preserved
    (``sort_keys=False``), but if the input is a hand-annotated config
    you care about, use ``ruamel.yaml`` instead.
    """
    if not isinstance(overrides, Mapping):
        raise TypeError(f"overrides must be a mapping, got {type(overrides).__name__}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level YAML in '{input_path}' must be a mapping (dict), "
            f"got {type(data).__name__}."
        )

    opts = _SetOptions(create_missing=create_missing)
    for k, v in overrides.items():
        if not isinstance(k, str):
            raise TypeError(f"Override keys must be strings, got {type(k).__name__}: {k!r}")
        if parse_values and isinstance(v, str):
            v = parse_value(v)
        set_by_path(data, k, v, opts)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    return data


# ---------------------------------------------------------------------------
# Thin CLI wrapper, so the command-line behaviour still routes through
# the same function.
# ---------------------------------------------------------------------------


def _parse_override_tokens(argv: List[str]) -> List[Tuple[str, str]]:
    """Parse unknown argv tokens into (key, raw_value). Supports --k=v and --k v."""
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise SystemExit(f"Unexpected token '{tok}'. Overrides must start with '--'.")
        tok = tok[2:]
        if "=" in tok:
            k, v = tok.split("=", 1)
            out.append((k, v))
            i += 1
            continue
        if i + 1 >= len(argv):
            raise SystemExit(f"Missing value for '--{tok}'. Use --{tok}=VALUE or --{tok} VALUE.")
        out.append((tok, argv[i + 1]))
        i += 2
    return out


def main(cmd_args: List[str] = None) -> int:
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Edit YAML by overriding existing keys from the command line.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_yaml", help="Input YAML file")
    parser.add_argument("output_yaml", help="Output YAML file")
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Allow creating missing keys along a path (default: error if key missing).",
    )
    args, unknown = parser.parse_known_args(cmd_args)

    tokens = _parse_override_tokens(unknown) if unknown else []
    # dict() would drop a repeated key; the last one wins either way, and
    # keeping insertion order matches the CLI's left-to-right semantics.
    overrides = {k: v for k, v in tokens}

    edit_yaml(
        args.input_yaml,
        args.output_yaml,
        overrides,
        create_missing=args.create_missing,
        parse_values=True,   # CLI values always arrive as strings
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
