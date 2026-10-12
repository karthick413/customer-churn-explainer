import sys
import importlib

print("Python:", sys.executable)
print("-" * 45)

packages = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "matplotlib",
    "seaborn",
    "streamlit",
    "joblib",
]

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"OK      {package}: {version}")
    except Exception as error:
        print(f"FAILED  {package}: {type(error).__name__}: {error}")

print("-" * 45)
print("Library check finished.")