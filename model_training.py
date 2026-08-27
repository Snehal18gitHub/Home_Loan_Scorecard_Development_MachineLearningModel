# ============================================================
# MODEL TRAINER
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


class ModelTrainer:

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Safe copies
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        # Store trained models
        self.models = {}

        # Store best tuned model
        self.best_model = None


    # ========================================================
    # TRAIN LOGISTIC REGRESSION
    # ========================================================

    def train_logistic_regression(self):

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Logistic Regression"] = model

        print(
            "Logistic Regression Training Completed"
        )

        return self


    # ========================================================
    # TRAIN DECISION TREE
    # ========================================================

    def train_decision_tree(self):

        model = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Decision Tree"] = model

        print(
            "Decision Tree Training Completed"
        )

        return self


    # ========================================================
    # TRAIN RANDOM FOREST
    # ========================================================

    def train_random_forest(self):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Random Forest"] = model

        print(
            "Random Forest Training Completed"
        )

        return self

    # ========================================================
    # HYPERPARAMETER TUNING - FASTER VERSION
    # ========================================================
    
    def hyperparameter_tuning(self):
    
        param_grid = {
    
            "n_estimators": [100, 200],
    
            "max_depth": [10, 20],
    
            "min_samples_split": [5],
    
            "min_samples_leaf": [2]
        }
    
        rf_model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=2
        )
    
        grid_search.fit(
            self.X_train,
            self.y_train
        )
    
        self.best_model = grid_search.best_estimator_
    
        self.models["Tuned Random Forest"] = (
            self.best_model
        )
    
        print("=" * 60)
        print("HYPERPARAMETER TUNING COMPLETED")
        print("=" * 60)
    
        print("\nBest Parameters:")
        print(grid_search.best_params_)
    
        print("\nBest Cross-Validation ROC-AUC:")
        print(round(grid_search.best_score_, 4))
    
        return self