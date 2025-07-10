#!/usr/bin/env python3
"""
Verification script for DSPO + SDXL project setup.

This script validates that all required files are present and have correct syntax.
Run this before starting your H100 training workflow.
"""

import py_compile
import sys
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()


def check_syntax(filepath: str) -> tuple[bool, str]:
    """Check Python file syntax."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as e:
        return False, str(e)


def main() -> int:
    """Main verification function."""
    project_root = Path(__file__).parent
    
    # Required files and directories
    required_structure = [
        "README.md",
        "requirements.txt", 
        "pyproject.toml",
        "Makefile",
        "data/prompts.csv",
        "data/annotations/ratings.csv",
        "dspo/__init__.py",
        "dspo/datasets.py",
        "dspo/reward.py", 
        "dspo/tuner.py",
        "scripts/generate_images.py",
        "scripts/build_pairs.py",
        "scripts/train_reward.py",
        "scripts/train_dspo.py",
        "scripts/sweep_dspo_optuna.py",
        "tests/test_dspo.py",
    ]
    
    # Python files to syntax check
    python_files = [
        "dspo/__init__.py",
        "dspo/datasets.py", 
        "dspo/reward.py",
        "dspo/tuner.py",
        "scripts/generate_images.py",
        "scripts/build_pairs.py", 
        "scripts/train_reward.py",
        "scripts/train_dspo.py",
        "scripts/sweep_dspo_optuna.py",
        "tests/test_dspo.py",
    ]
    
    print("🔍 DSPO + SDXL Project Verification")
    print("=" * 50)
    
    # Check file structure
    print("\n📁 Checking project structure...")
    missing_files = []
    for file_path in required_structure:
        full_path = project_root / file_path
        if check_file_exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    # Check Python syntax
    print("\n🐍 Checking Python syntax...")
    syntax_errors = []
    for py_file in python_files:
        full_path = project_root / py_file
        if check_file_exists(full_path):
            is_valid, message = check_syntax(full_path)
            if is_valid:
                print(f"✅ {py_file}")
            else:
                print(f"❌ {py_file} - SYNTAX ERROR: {message}")
                syntax_errors.append((py_file, message))
        else:
            print(f"⚠️  {py_file} - FILE NOT FOUND")
    
    # Summary
    print("\n📊 Verification Summary")
    print("=" * 30)
    
    if not missing_files and not syntax_errors:
        print("🎉 ALL CHECKS PASSED!")
        print("\n🚀 Your DSPO + SDXL project is ready for H100 training!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the full pipeline: make all")
        print("3. Or start with image generation: make generate")
        return 0
    else:
        print("❌ ISSUES FOUND:")
        if missing_files:
            print(f"   - {len(missing_files)} missing files")
        if syntax_errors:
            print(f"   - {len(syntax_errors)} syntax errors")
        print("\nPlease fix these issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
