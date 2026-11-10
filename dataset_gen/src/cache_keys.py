#!/usr/bin/env python3
"""
One recipe for the keys of the project's on-disk caches (T-19, 2026-10-01). A cached value is a pure function of

  - its input FILES, each identified by (resolved path, size, mtime_ns) (`file_identity`): a file rewritten in place
    (scripts/revoxelize_parity.py and scripts/add_voxel_noise.py rewrite a sample's .bin) gets a new mtime, so every value
    derived from it misses;
  - the VOXELIZER, where the cached step voxelizes by itself (reco_engine/cache.Cache.voxel_key: STL -> grid). A cache that
    reads a stored grid (a sample's .bin) names none: the grid file's identity stands for the voxelizer that wrote it;
  - the CODE that computes it (`code_fingerprint`): per module, a hash of its source with comments, docstrings and
    formatting removed (ast.unparse), plus the live values of its UPPER_CASE constants. An edit of the encoder or of one of
    its constants changes the key, a docstring or comment edit does not;
  - its parameters (resolution, truncation, ...).

    key = cache_key("voxfield", files=[bin, inside, npz], code=["training.inputs", "src.field_channels"],
                    params=dict(resolution=64, saturation=16.0))

The source digest of a module is taken once per process (the first time a key needs it): the code a process runs is the
code it imported, so an edit during a long run does not relabel the values the old code goes on computing. Callers in
tests that rewrite a module call `source_digest.cache_clear()`.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional

REPO = Path(__file__).resolve().parents[2]
_SIMPLE = (bool, int, float, str, bytes, type(None))


def _importable() -> None:
    """Module names of both conventions resolve: "training.inputs" (repo root) and "src.field_channels" (dataset_gen)."""
    for root in (REPO, REPO / "dataset_gen"):
        if str(root) not in sys.path:
            sys.path.append(str(root))


def file_identity(path) -> str:
    """"<resolved path>:<size>:<mtime_ns>" of an existing file (raises FileNotFoundError otherwise)."""
    p = Path(path).resolve()
    st = p.stat()
    return f"{p}:{st.st_size}:{st.st_mtime_ns}"


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


@functools.lru_cache(maxsize=None)
def source_digest(path: str) -> str:
    """sha1 of a Python file's code: parsed, docstrings dropped, unparsed (so comments and formatting do not count)."""
    tree = _strip_docstrings(ast.parse(Path(path).read_text(encoding="utf-8")))
    return hashlib.sha1(ast.unparse(tree).encode()).hexdigest()


def _plain(value) -> bool:
    """A scalar, or tuples/lists/dicts of them: its repr is the same in every process."""
    if isinstance(value, _SIMPLE):
        return True
    if isinstance(value, (tuple, list)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, _SIMPLE) and _plain(v) for k, v in value.items())
    return False


def _simple(value) -> bool:
    """`_plain`, or a set of scalars (written sorted by `_canonical`)."""
    if isinstance(value, (set, frozenset)):
        return all(isinstance(v, _SIMPLE) for v in value)
    return _plain(value)


def _canonical(value) -> str:
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(repr(v) for v in value)) + "}"
    return repr(value)


def module_constants(module) -> str:
    """The module's UPPER_CASE globals with plain values (numbers, strings, tuples/lists/dicts of them), sorted, as text."""
    items = sorted(((k, v) for k, v in vars(module).items() if k.isupper() and not k.startswith("_") and _simple(v)),
                   key=lambda kv: kv[0])
    return ";".join(f"{k}={_canonical(v)}" for k, v in items)


def _module(name_or_module):
    if isinstance(name_or_module, str):
        _importable()
        return importlib.import_module(name_or_module)
    return name_or_module


def code_fingerprint(*modules) -> str:
    """sha1 over the given modules (module objects or importable names; order does not matter): each module's
    `source_digest` (its file name relative to the repository, so every checkout of the same code agrees) and its
    `module_constants` (read live)."""
    parts = []
    for m in modules:
        mod = _module(m)
        path = Path(mod.__file__).resolve()
        try:
            name = str(path.relative_to(REPO))
        except ValueError:
            name = path.name
        parts.append(f"{name}#{source_digest(str(path))}#{module_constants(mod)}")
    return hashlib.sha1("\n".join(sorted(parts)).encode()).hexdigest()


def cache_key(cache: str, *, files: Iterable = (), voxelizer: Optional[str] = None, code: Iterable = (),
              params: Optional[Mapping] = None, length: Optional[int] = None) -> str:
    """The key of one cache entry (hex sha1, `length` characters if given): the cache's name, `file_identity` of every
    input file (order kept), the voxelizer (None: the inputs are stored grids, see the module docstring), the
    `code_fingerprint` of the modules in `code`, and the parameters (JSON, sorted keys)."""
    parts = [f"cache={cache}"]
    parts += [f"file={file_identity(f)}" for f in files]
    parts.append(f"voxelizer={voxelizer if voxelizer is not None else '-'}")
    code = list(code)
    parts.append(f"code={code_fingerprint(*code) if code else '-'}")
    parts.append("params=" + json.dumps(dict(params or {}), sort_keys=True, default=repr))
    h = hashlib.sha1()
    for part in parts:
        b = part.encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    digest = h.hexdigest()
    return digest[:length] if length else digest


def cache_root(subdir: str) -> Path:
    """$SHAPE_RECO_CACHE or ~/.cache/shape_reco, then `subdir` (the training-side caches' common root)."""
    return Path(os.environ.get("SHAPE_RECO_CACHE", Path.home() / ".cache/shape_reco")) / subdir
