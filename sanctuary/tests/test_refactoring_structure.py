"""
Test the structure of the refactored core module.

This test verifies that:
1. All expected files exist
2. All files are under 12KB
3. The module structure is correct
"""

import os
import sys


def test_file_existence():
    """Test that all expected files exist."""
    core_path = 'sanctuary/mind/cognitive_core/core'
    
    expected_files = [
        '__init__.py',
        'action_executor.py',
        'cognitive_loop.py',
        'cycle_executor.py',
        'lifecycle.py',
        'state_manager.py',
        'subsystem_coordinator.py',
        'timing.py'
    ]
    
    print("Checking file existence:")
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(core_path, filename)
        if os.path.exists(filepath):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_file_sizes():
    """Test that all files are under 12KB."""
    core_path = 'sanctuary/mind/cognitive_core/core'
    
    files = [f for f in os.listdir(core_path) if f.endswith('.py')]
    
    print("\nChecking file sizes (must be < 12KB):")
    all_under_limit = True
    for filename in sorted(files):
        filepath = os.path.join(core_path, filename)
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        
        if size_kb < 12:
            print(f"  ✅ {filename:30s} {size_kb:5.1f} KB")
        else:
            print(f"  ❌ {filename:30s} {size_kb:5.1f} KB - EXCEEDS LIMIT")
            all_under_limit = False
    
    return all_under_limit


def test_module_count():
    """Test that we have the expected number of modules."""
    core_path = 'sanctuary/mind/cognitive_core/core'
    
    files = [f for f in os.listdir(core_path) if f.endswith('.py')]
    
    expected_count = 8
    actual_count = len(files)
    
    print(f"\nModule count: {actual_count} (expected: {expected_count})")
    
    if actual_count == expected_count:
        print(f"  ✅ Correct number of modules")
        return True
    else:
        print(f"  ❌ Expected {expected_count} modules, found {actual_count}")
        return False


def test_legacy_file_moved():
    """Test that the old core.py has been renamed to core_legacy.py."""
    old_path = 'sanctuary/mind/cognitive_core/core.py'
    legacy_path = 'sanctuary/mind/cognitive_core/core_legacy.py'
    
    print("\nChecking legacy file:")
    
    if os.path.exists(old_path):
        print(f"  ❌ Old core.py still exists (should be renamed)")
        return False
    
    if os.path.exists(legacy_path):
        print(f"  ✅ core_legacy.py exists (old file backed up)")
        size = os.path.getsize(legacy_path)
        print(f"  ✅ core_legacy.py size: {size / 1024:.1f} KB (~57KB expected)")
        return True
    
    print(f"  ⚠️  Neither core.py nor core_legacy.py found")
    return False


def test_imports_syntax():
    """Test that all module files have valid Python syntax."""
    core_path = 'sanctuary/mind/cognitive_core/core'
    
    files = [f for f in os.listdir(core_path) if f.endswith('.py')]
    
    print("\nChecking Python syntax:")
    all_valid = True
    for filename in sorted(files):
        filepath = os.path.join(core_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), filepath, 'exec')
            print(f"  ✅ {filename}")
        except SyntaxError as e:
            print(f"  ❌ {filename} - SYNTAX ERROR: {e}")
            all_valid = False

    return all_valid


def test_single_responsibility():
    """Test that each module has a clear, documented responsibility."""
    core_path = 'sanctuary/mind/cognitive_core/core'

    files = [f for f in os.listdir(core_path) if f.endswith('.py') and f != '__init__.py']

    print("\nChecking module docstrings:")
    all_documented = True
    for filename in sorted(files):
        filepath = os.path.join(core_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check if file starts with docstring
            if content.strip().startswith('"""') or content.strip().startswith("'''"):
                # Extract first line of docstring
                lines = content.split('\n')
                docstring_line = None
                for line in lines[1:10]:  # Check first 10 lines
                    if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                        docstring_line = line.strip()
                        break
                
                if docstring_line:
                    print(f"  ✅ {filename:30s} {docstring_line[:40]}...")
                else:
                    print(f"  ⚠️  {filename:30s} Has docstring but unclear")
            else:
                print(f"  ❌ {filename:30s} Missing module docstring")
                all_documented = False
    
    return all_documented


def main():
    """Run all structure tests."""
    print("=" * 70)
    print("Testing Refactored Core Module Structure")
    print("=" * 70)
    print()
    
    tests = [
        ("File existence", test_file_existence),
        ("File sizes (<12KB)", test_file_sizes),
        ("Module count", test_module_count),
        ("Legacy file moved", test_legacy_file_moved),
        ("Python syntax", test_imports_syntax),
        ("Module documentation", test_single_responsibility),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}")
        result = test_func()
        results.append((name, result))
    
    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All structure tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
