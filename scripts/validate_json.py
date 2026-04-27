import json
import os
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

    return errors


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


def is_protocol_file(filepath):
    """Check if a file is a protocol file based on its path and name."""
    return 'Protocols' in filepath or '_protocol' in os.path.basename(filepath).lower()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate JSON data files against schemas."
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
        schema_file = None
        
        # Handle special cases and protocols
        if is_protocol_file(json_file):
            schema_file = os.path.join(schemas_dir, "memory_protocol.schema.json")
        elif base == "journal_index.json":
            schema_file = os.path.join(schemas_dir, "journal_index.schema.json")
        elif base == "journal_manifest.json":
            schema_file = os.path.join(schemas_dir, "journal_manifest.schema.json")
        elif base == "trace_archive.json":
            schema_file = os.path.join(schemas_dir, "trace_archive.schema.json")
        # Journal entries
        elif os.path.dirname(json_file).endswith("journal"):
            schema_file = os.path.join(schemas_dir, "journal_entry.schema.json")
            errors = validate_journal(schema_file, json_file)
        else:
            # Try to find matching schema
            name, _ = os.path.splitext(base)
            candidate = os.path.join(schemas_dir, f"{name}.schema.json")
            if os.path.exists(candidate):
                schema_file = candidate
            else:
                # Try by directory name
                dir_name = os.path.basename(os.path.dirname(json_file))
                candidate = os.path.join(schemas_dir, f"{dir_name}.schema.json")
                if os.path.exists(candidate):
                    schema_file = candidate

        if not schema_file:
            print(f"SKIPPED: No schema found for {json_file}")
            summary["skipped"] += 1
            continue

        if not os.path.dirname(json_file).endswith("journal"):
            errors = validate_json(schema_file, json_file)

        if errors:
            print(f"VALIDATION ERRORS in {json_file} against {schema_file}:")
            for err in errors:
                print(f"  - {err}")
            summary["errors"] += 1
        else:
            print(f"SUCCESS: {json_file} conforms to {schema_file}")
            summary["validated"] += 1

    print(f"\nValidation complete.\n  Validated: {summary['validated']}\n  Errors: {summary['errors']}\n  Skipped (no schema): {summary['skipped']}")