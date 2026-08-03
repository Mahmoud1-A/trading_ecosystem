"""Patch the 2020-2022 materializer to discover every local catalog copy."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "build_multiyear_usa500_dataset.py"


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"expected block not found:\n{old[:300]}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    old_catalog_paths = '''def _catalog_paths() -> list[Path]:
    paths = sorted(ARCHIVE_ROOT.rglob("dataset_catalog.json"))
    if MAIN_CATALOG_PATH.is_file() and MAIN_CATALOG_PATH not in paths:
        paths.insert(0, MAIN_CATALOG_PATH)
    return paths
'''
    new_catalog_paths = '''def _catalog_paths() -> list[Path]:
    """Find archive and control-plane catalog copies without scanning .venv."""
    candidates: set[Path] = set()
    roots = (
        ARCHIVE_ROOT,
        FRAMEWORK_ROOT / "artifacts",
        REPO_ROOT / "artifacts",
        FRAMEWORK_ROOT,
    )
    ignored = {".git", ".venv", "venv", "site-packages", "node_modules", "__pycache__"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("dataset_catalog.json"):
            if any(part in ignored for part in path.parts):
                continue
            candidates.add(path.resolve())
    if MAIN_CATALOG_PATH.is_file():
        candidates.add(MAIN_CATALOG_PATH.resolve())
    return sorted(candidates, key=lambda p: (0 if p == MAIN_CATALOG_PATH.resolve() else 1, str(p)))


def _iter_dict_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dict_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dict_nodes(child)


def _catalog_targets() -> list[tuple[Path, Any, dict[str, Any]]]:
    """Return every catalog copy containing the canonical 2024 USA500 entry.

    A fallback USA500/US500 CFD entry is accepted only when the exact canonical
    ID is absent everywhere. This keeps the builder usable with copied/renamed
    control-plane catalogs while preserving the entry schema.
    """
    exact: list[tuple[Path, Any, dict[str, Any]]] = []
    fallback: list[tuple[Path, Any, dict[str, Any]]] = []
    inspected: list[str] = []
    for catalog_path in _catalog_paths():
        inspected.append(str(catalog_path))
        try:
            doc = _load_json(catalog_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping unreadable catalog {catalog_path}: {exc}")
            continue
        slot = _find_entry_slot(doc, TEMPLATE_DATASET_ID)
        if slot is not None:
            exact.append((catalog_path, doc, slot[2]))
            continue
        for node in _iter_dict_nodes(doc):
            blob = json.dumps(node, default=str).lower()
            dataset_id = str(node.get("dataset_id") or node.get("id") or "")
            if dataset_id and ("usa500" in blob or "us500" in blob) and "cfd" in blob:
                fallback.append((catalog_path, doc, node))
                break
    if exact:
        return exact
    if fallback:
        print(
            "WARNING: canonical 2024 template ID was not found; using compatible "
            f"USA500 CFD catalog entry {fallback[0][2].get('dataset_id') or fallback[0][2].get('id')}"
        )
        return fallback
    raise BuildError(
        "no compatible USA500 CFD catalog template found; inspected=" + repr(inspected)
    )
'''
    text = replace_once(text, old_catalog_paths, new_catalog_paths)

    old_build_start = '''def build(*, overwrite: bool = False, dry_run: bool = False) -> Path:
    if not MAIN_CATALOG_PATH.is_file():
        raise BuildError(f"main catalog missing: {MAIN_CATALOG_PATH}")
    catalog_doc = _load_json(MAIN_CATALOG_PATH)
    template_slot = _find_entry_slot(catalog_doc, TEMPLATE_DATASET_ID)
    if template_slot is None:
        raise BuildError(f"template catalog entry missing: {TEMPLATE_DATASET_ID}")
    _parent, _slot, template = template_slot

    yearly = [_resolve_year(year, SOURCE_IDS[year]) for year in TARGET_YEARS]
'''
    new_build_start = '''def build(*, overwrite: bool = False, dry_run: bool = False) -> Path:
    catalog_targets = _catalog_targets()
    primary_catalog_path, _primary_doc, template = catalog_targets[0]
    catalog_root = primary_catalog_path.parent
    print(f"Using catalog template from: {primary_catalog_path}")

    yearly = [_resolve_year(year, SOURCE_IDS[year]) for year in TARGET_YEARS]
'''
    text = replace_once(text, old_build_start, new_build_start)

    text = replace_once(
        text,
        '''    out_dir = (
        MAIN_CATALOG_ROOT
''',
        '''    out_dir = (
        catalog_root
''',
    )

    old_registration = '''    _remove_existing_entry(catalog_doc, TARGET_DATASET_ID)
    template_slot = _find_entry_slot(catalog_doc, TEMPLATE_DATASET_ID)
    if template_slot is None:
        raise BuildError("template entry disappeared while updating catalog")
    parent, slot, _template = template_slot
    if isinstance(parent, list):
        parent.append(new_entry)
    elif isinstance(parent, dict) and slot == TEMPLATE_DATASET_ID:
        parent[TARGET_DATASET_ID] = new_entry
    elif isinstance(parent, dict):
        # Template is stored as a normal node inside a list-like dict field.
        container = parent.get(slot)
        if isinstance(container, list):
            container.append(new_entry)
        else:
            # Most catalog layouts use a list or a dataset-id keyed map. Keep a
            # deterministic side collection only as a final compatible fallback.
            parent.setdefault("multiyear_entries", []).append(new_entry)
    else:
        raise BuildError("unsupported catalog container")
    _write_json_atomic(MAIN_CATALOG_PATH, catalog_doc)

    print(f"Registered {TARGET_DATASET_ID}")
'''
    new_registration = '''    registered_paths: list[str] = []
    for catalog_path, catalog_doc, catalog_template in catalog_targets:
        _remove_existing_entry(catalog_doc, TARGET_DATASET_ID)
        template_id = str(
            catalog_template.get("dataset_id")
            or catalog_template.get("id")
            or TEMPLATE_DATASET_ID
        )
        template_slot = _find_entry_slot(catalog_doc, template_id)
        if template_slot is None:
            raise BuildError(
                f"template entry {template_id!r} disappeared while updating {catalog_path}"
            )
        parent, slot, _template = template_slot
        entry_copy = copy.deepcopy(new_entry)
        if isinstance(parent, list):
            parent.append(entry_copy)
        elif isinstance(parent, dict) and slot == template_id:
            parent[TARGET_DATASET_ID] = entry_copy
        elif isinstance(parent, dict):
            container = parent.get(slot)
            if isinstance(container, list):
                container.append(entry_copy)
            else:
                parent.setdefault("multiyear_entries", []).append(entry_copy)
        else:
            raise BuildError(f"unsupported catalog container in {catalog_path}")
        _write_json_atomic(catalog_path, catalog_doc)
        registered_paths.append(str(catalog_path))

    print(f"Registered {TARGET_DATASET_ID} in {len(registered_paths)} catalog(s)")
    for registered in registered_paths:
        print(f"Catalog: {registered}")
'''
    text = replace_once(text, old_registration, new_registration)

    TARGET.write_text(text, encoding="utf-8")
    print("Multiyear catalog discovery hardened.")


if __name__ == "__main__":
    main()
