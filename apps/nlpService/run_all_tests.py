import importlib.util
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "testing"

def run_test_file(file_path):
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "main"):
        module.main()

def run_all_tests():
    print("🚀 Running all NLP service tests...\n" + "="*40)
    for file in TEST_DIR.glob("test_*.py"):
        print(f"\n🧪 Running {file.name} ...")
        try:
            run_test_file(file)
            print("   ✅ Success")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    run_all_tests()
