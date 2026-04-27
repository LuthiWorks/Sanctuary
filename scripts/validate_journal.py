import json
import sys
from jsonschema import Draft202012Validator


def validate_journal(schema_path, data_path):
    # Load schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the subschema for a single journal entry
    entry_schema = schema["items"]["properties"]["journal_entry"]
    validator = Draft202012Validator(entry_schema)
    errors = []

    # Validate each journal_entry object in the journal array
    for idx, entry in enumerate(data):
        journal_obj = entry.get("journal_entry")
        if not journal_obj:
            errors.append(f"Entry {idx}: Missing 'journal_entry' object.")
            continue
        for error in validator.iter_errors(journal_obj):
            errors.append(f"Entry {idx}: {error.message}")

    # Report results
    if errors:
        print(f"VALIDATION ERRORS in {data_path} against {schema_path}:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"SUCCESS: {data_path} conforms to {schema_path}")


import os
import glob

def find_schema_for_json(json_path, schemas_dir):
    """Find the best matching schema for a given JSON file."""
    base = os.path.basename(json_path)
    name, _ = os.path.splitext(base)
    # Try exact match
    candidate = os.path.join(schemas_dir, f"{name}.schema.json")
    if os.path.exists(candidate):
        return candidate
    # Try manifest/index/entry suffixes
    for suffix in ["_manifest", "_index", "_entry", "_archive", "_log", "_protocol", "_glyphs", "_directives"]:
        candidate = os.path.join(schemas_dir, f"{name}{suffix}.schema.json")
        if os.path.exists(candidate):
            return candidate
    # Try matching by directory name
    dir_name = os.path.basename(os.path.dirname(json_path))
    candidate = os.path.join(schemas_dir, f"{dir_name}.schema.json")
    if os.path.exists(candidate):
        return candidate
    return None

def validate_json(schema_path, data_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    validator = Draft202012Validator(schema)
    errors = []
    for error in validator.iter_errors(data):
        errors.append(error.message)
    return errors


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate journal/JSON data files against schemas."
    )
    parser.add_argument(
        "--data-root",
        default=str(repo_default / "sanctuary" / "data"),
        help="Directory containing JSON files to validate.",
    )
    parser.add_argument(
        "--schemas-dir",
        default=str(repo_default / "schemas"),
        help="Directory containing schema files.",
    )
    args = parser.parse_args()
    repo_root = args.data_root
    schemas_dir = args.schemas_dir

    all_json_files = []
    for root, dirs, files in os.walk(repo_root):
        for file in files:
            if file.endswith(".json"):
                all_json_files.append(os.path.join(root, file))

    print(f"Validating {len(all_json_files)} JSON files against schemas in {schemas_dir}\n")
    summary = {"validated": 0, "errors": 0, "skipped": 0}
    for json_file in all_json_files:
        base = os.path.basename(json_file)
        # Use dedicated schemas for special files
        if base == "journal_index.json":
            schema_file = os.path.join(schemas_dir, "journal_index.schema.json")
        elif base == "journal_manifest.json":
            schema_file = os.path.join(schemas_dir, "journal_manifest.schema.json")
        elif base == "trace_archive.json":
            schema_file = os.path.join(schemas_dir, "trace_archive.schema.json")
        # Use journal_entry schema for journal entry files
        elif os.path.dirname(json_file).endswith("journal"):
            schema_file = os.path.join(schemas_dir, "journal_entry.schema.json")
        else:
            schema_file = find_schema_for_json(json_file, schemas_dir)
        if not schema_file:
            print(f"SKIPPED: No schema found for {json_file}")
            summary["skipped"] += 1
            continue
        try:
            errors = validate_json(schema_file, json_file)
            if errors:
                print(f"VALIDATION ERRORS in {json_file} against {schema_file}:")
                for err in errors:
                    print(f"  - {err}")
                summary["errors"] += 1
            else:
                print(f"SUCCESS: {json_file} conforms to {schema_file}")
                summary["validated"] += 1
        except Exception as e:
            print(f"ERROR: Could not validate {json_file}: {e}")
            summary["errors"] += 1

    print(f"\nValidation complete.")
    print(f"  Validated: {summary['validated']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Skipped (no schema): {summary['skipped']}")
