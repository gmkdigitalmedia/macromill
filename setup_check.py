"""
Setup Verification Script
Checks if all dependencies and requirements are properly installed
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"[PASS] Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"[FAIL] Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        "torch",
        "transformers",
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn"
    ]

    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"[PASS] {package}")
        except ImportError:
            print(f"[FAIL] {package} (missing)")
            all_ok = False

    return all_ok


def check_files():
    """Check if required files exist"""
    print("\nChecking project files...")
    required_files = [
        "train.py",
        "app.py",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "README.md"
    ]

    all_ok = True
    for file in required_files:
        if Path(file).exists():
            print(f"[PASS] {file}")
        else:
            print(f"[FAIL] {file} (missing)")
            all_ok = False

    return all_ok


def check_dataset():
    """Check if dataset exists"""
    print("\nChecking dataset...")
    dataset_path = Path("archive/IMDB Dataset.csv")

    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"[PASS] Dataset found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"[FAIL] Dataset not found at {dataset_path}")
        return False


def check_docker():
    """Check if Docker is available"""
    print("\nChecking Docker...")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[PASS] {version}")
            return True
        else:
            print("[FAIL] Docker not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[FAIL] Docker not installed")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA support...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[PASS] CUDA available ({device_name})")
            return True
        else:
            print("WARNING: CUDA not available (will use CPU)")
            return True  # Not critical
    except Exception as e:
        print(f"WARNING: Could not check CUDA: {str(e)}")
        return True  # Not critical


def main():
    """Run all checks"""
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Files", check_files),
        ("Dataset", check_dataset),
        ("Docker", check_docker),
        ("CUDA", check_cuda)
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"[FAIL] {check_name} check failed: {str(e)}")
            results[check_name] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    critical_checks = ["Python Version", "Dependencies", "Project Files", "Dataset"]
    critical_passed = all(results.get(check, False) for check in critical_checks)

    for check_name, result in results.items():
        status = "[PASS] OK" if result else "[FAIL] FAIL"
        print(f"{check_name}: {status}")

    if critical_passed:
        print("\n[PASS] All critical checks passed - Ready to proceed!")
        print("\nNext steps:")
        print("1. Train the model: python train.py")
        print("2. Start the API: python app.py")
        print("3. Or use Docker: docker-compose up -d")
        return 0
    else:
        print("\n[FAIL] Some critical checks failed - Please fix before proceeding")
        print("\nTo install dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit(main())
