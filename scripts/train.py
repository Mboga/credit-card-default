# mlflow for MLOPS
import os
import mlflow
import xgboost as xgb
from pathlib import Path
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from mlflow.models.signature import infer_signature
from scripts.processing import data_loader,prepare_model_data,extract_features


#use logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configuration and paths 

# Flexible pathing
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"
EXPERIMENT_NAME = "Credit_default_XGBoost_Prod"


# Use the environment variable if available (set by Docker), otherwise fallback to local for testing
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


#xgb tuned parameters
XGB_PARAMS = {
    "n_estimators":500, 
    "learning_rate":0.08, 
    "scale_pos_weight":3.52,#23364/6636
    "tree_method":'exact', 
    "n_jobs":1, 
    "random_state":42
}

def run_training():
    """
    Main pipeline training logic

    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logging.info(f"ML flow tracking at :{TRACKING_URI}")

    # Load data

    try:
        train, val, test = data_loader(DATA_PATH)
        X_train, y_train = prepare_model_data(train)
        X_val, y_val = prepare_model_data(val)
        X_test, y_test = prepare_model_data(test)
    except Exception as e:
        logging.error(f"Data Loading Failed: {e}")

    # Final validation before training starts: # preflight check
    assert not np.isinf(X_train).values.any(), "Inf detected in features!"
    assert not X_train.isnull().values.any(), "NaN detected in features!"
    logging.info("Data is numerically stable. Ready for XGBoost.")


    with mlflow.start_run(run_name="XGB_Production_v1"):

        # Log Parameters
        mlflow.log_params(XGB_PARAMS)

        #Train Model
        model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)],
            verbose=False
        )

        #Logging model with signature

        X_sample = X_train.astype(float) # to avoid integer warning at inference
        signature = infer_signature(X_sample,model.predict(X_sample))

        # Create an input sample
        input_sample = X_sample.iloc[[0]]

        # Save the model and the signature
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_sample,
            registered_model_name="CreditDefault_XGB"
        )
        
        # Log validation metrics
        val_auc = model.best_score
        mlflow.log_metric("val_auc", val_auc)

        #Save the feature list
        feature_list = X_train.columns.tolist()
        mlflow.log_dict(feature_list,"feature_list.json")

        # get prediction probabilities
        test_probs = model.predict_proba(X_test)[:,1]

        # calculate the standard ROC-AUC
        test_roc_auc = roc_auc_score (y_test, test_probs)
        mlflow.log_metric("test_roc_auc",test_roc_auc)

        # calculate the PR-AUC (Average Precision)
        test_pr_auc = average_precision_score(y_test,test_probs)
        mlflow.log_metric("test_pr_auc",test_pr_auc)

        print(f"ROC-AUC:{test_roc_auc:.4f}")
        print(f"PR-AUC:{test_pr_auc:.4f}")

        # Save the PR curve plot as an  MLflow Artifact
        precision, recall, _ = precision_recall_curve(y_test, test_probs)

        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label=f"PR Curve (AUC={test_pr_auc: .2f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")
        plt.close()


        # calculate the FPR, TPR, and Thresholds
        fpr, tpr, thresholds = roc_curve(y_test,test_probs)
        # test_roc_auc already logged
        # create chart
        plt.figure (figsize=(8,6))
        plt.plot(fpr,tpr,color='darkorange',lw=2, label=f"AUC ={test_roc_auc:.2f}")
        plt.plot([0,1],[0,1], color='navy',lw=2,linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        #log the thresholds
        thresh_df = pd.DataFrame({'threshold': thresholds,'fpr':fpr, 'tpr':tpr})
        thresh_df.to_csv("roc_threshold_data.csv", index=False)
        mlflow.log_artifact("roc_threshold_data.csv")


        logging.info(f"Training Complete")


        # sanity check
        run_id = mlflow.active_run().info.run_id
        test_uri = f"runs:/{run_id}/model"
        check_model = mlflow.xgboost.load_model(test_uri)
        sample = X_test.iloc[[0]].astype(float)

        logging.info(f"Sanity Check Probability: {check_model.predict_proba(sample)[:,1][0]:.4f}")


if __name__ == "__main__":
    run_training()




