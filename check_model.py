from pathlib import Path
import pickle


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


print("=" * 60)
print("CHECKING MODEL FILES")
print("=" * 60)

print("\nModel file:")
print(MODEL_PATH)
print("Exists:", MODEL_PATH.exists())

print("\nFeature names file:")
print(FEATURE_PATH)
print("Exists:", FEATURE_PATH.exists())


# ============================================================
# CHECK MODEL FILE
# ============================================================

if MODEL_PATH.exists():

    # --------------------------------------------------------
    # TRY JOBLIB
    # --------------------------------------------------------

    try:

        import joblib

        model = joblib.load(MODEL_PATH)

        print("\nMODEL LOAD WITH JOBLIB: SUCCESS")
        print("Model type:", type(model))

    except Exception as e:

        print("\nMODEL LOAD WITH JOBLIB: FAILED")
        print("Error:", repr(e))


# ============================================================
# CHECK FEATURE NAMES FILE
# ============================================================

if FEATURE_PATH.exists():

    try:

        with open(FEATURE_PATH, "rb") as file:
            feature_names = pickle.load(file)

        print("\nFEATURE NAMES LOAD: SUCCESS")
        print("Feature type:", type(feature_names))

        if hasattr(feature_names, "__len__"):
            print("Number of features:", len(feature_names))

    except Exception as e:

        print("\nFEATURE NAMES LOAD: FAILED")
        print("Error:", repr(e))