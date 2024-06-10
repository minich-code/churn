
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import os
from src.churn import logging
from src.churn.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, X_train_transformed, X_val_transformed, y_train, y_val):
        lgbm_model = LGBMClassifier(
            boosting_type=self.config.boosting_type,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            objective=self.config.objective,
            min_split_gain=self.config.min_split_gain,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            min_child_samples=self.config.min_child_samples,
            verbose=0,
            force_row_wise=True
        )

        # Initialize the Stratified K-Fold cross-validator
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform Stratified K-Fold Cross-Validation
        fold_f1_scores = []
        for train_index, val_index in skf.split(X_train_transformed, y_train):
            X_train_fold, X_val_fold = X_train_transformed[train_index], X_train_transformed[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # Train the model on the training fold
            lgbm_model.fit(X_train_fold, y_train_fold)
            
            # Validate the model on the validation fold
            y_val_pred = lgbm_model.predict(X_val_fold)
            
            # Evaluate the model
            fold_f1 = f1_score(y_val_fold, y_val_pred, average='macro')
            fold_f1_scores.append(fold_f1)
            print(f"Fold Validation Macro F1-Score: {fold_f1}")
        
        # Print average Macro F1-Score across all folds
        print(f"Average Cross-Validation Macro F1-Score: {sum(fold_f1_scores) / len(fold_f1_scores)}")

        # Final training on full training set
        lgbm_model.fit(X_train_transformed, y_train)
        
        joblib.dump(lgbm_model, os.path.join(self.config.root_dir, self.config.model_name))
        # Logging info 
        logging.info(f"Model Trainer completed: Saved to {os.path.join(self.config.root_dir, self.config.model_name)}")

        # # Validate on the separate validation set
        # y_val_pred = lgbm_model.predict(X_val_transformed)
        # print(f"Validation Set Classification Report:\n {classification_report(y_val, y_val_pred)}")
        # print(f"Validation Set Confusion Matrix:\n {confusion_matrix(y_val, y_val_pred)}")
