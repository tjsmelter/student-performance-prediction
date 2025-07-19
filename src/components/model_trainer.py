import os
import sys
from dataclasses import dataclass

import pandas as pd

import shap
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

#added this
from src.components.model_trainer_config import ModelTrainerConfig

from src.utils import save_object,evaluate_models
from src.utils import load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Predict test set
            predicted = best_model.predict(X_test)

            # ✅ Evaluate model performance
            r2_square = r2_score(y_test, predicted)

            # Load preprocessor to get feature names
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            preprocessor = load_object(preprocessor_path)

            # Define your columns explicitly
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
      
            try:
                sample_X = X_test[:100] if X_test.shape[0] > 100 else X_test

                if best_model_name == "CatBoosting Regressor":
                    # CatBoost needs original feature names, no one-hot encoding
                    feature_names = numerical_columns + categorical_columns
                    
                    # Reconstruct DataFrame from X_test with original column names
                    sample_X_df = pd.DataFrame(sample_X, columns=feature_names)
                    
                else:
                    # For XGBRegressor and others using one-hot encoded features
                    # Use preprocessor to get one-hot encoded feature names
                    cat_pipeline = preprocessor.named_transformers_["cat_pipelines"]
                    ohe = cat_pipeline.named_steps["one_hot_encoder"]
                    cat_feature_names = ohe.get_feature_names_out(categorical_columns)
                    feature_names = numerical_columns + list(cat_feature_names)
                    
                    # Reconstruct DataFrame with expanded feature names
                    sample_X_df = pd.DataFrame(sample_X, columns=feature_names)
                    
                logging.info(f"sample_X_df.shape: {sample_X_df.shape}")
                logging.info(f"sample_X_df.columns: {sample_X_df.columns.tolist()[:5]}")
       

            except Exception as e:
                logging.warning(f"Could not reconstruct feature names: {e}")
                feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
                sample_X_df = pd.DataFrame(sample_X, columns=feature_names)

            # Then SHAP explainability block
            try:
                logging.info("Generating SHAP explainability plot")
                print(best_model_name)
                if best_model_name in ["Linear Regression", "XGBRegressor", "CatBoosting Regressor", "Random Forest", "Gradient Boosting"]:
                    
                    logging.info(f"sample_X_df shape: {sample_X_df.shape}")
                    logging.info(f"sample_X_df columns: {sample_X_df.columns.tolist()}")
                    logging.info(f"sample_X_df head:\n{sample_X_df.head()}")    

                    explainer = shap.Explainer(best_model, sample_X_df, feature_names=feature_names)
                    shap_values = explainer(sample_X_df)

                    # ✅ Add this to generate waterfall plot
                    instance_index = 0
                    shap_value_single = shap_values[instance_index]
                    shap.plots.waterfall(shap_value_single, show=False)
                    plt.title(f"SHAP Waterfall Plot - Row {instance_index} - {best_model_name}")
                    plt.tight_layout()
                    plt.savefig(f"artifacts/shap_waterfall_plot_row_{instance_index}.png")
                    plt.close()
                    logging.info(f"SHAP waterfall plot saved to artifacts/shap_waterfall_plot_row_{instance_index}.png")

                    shap.summary_plot(shap_values, sample_X_df, show=False)
                    plt.title(f"SHAP Summary Plot - {best_model_name}")
                    plt.tight_layout()
                    os.makedirs("artifacts", exist_ok=True)
                    plt.savefig("artifacts/shap_summary_plot.png")
                    plt.close()
                    logging.info("SHAP summary plot saved to artifacts/shap_summary_plot.png")

                    # Pick a single row to explain (e.g., the first one)
                    instance_index = 0  # you can change this to another index
                    shap_value_single = shap_values[instance_index]

                    # Plot waterfall
                    shap.plots.waterfall(shap_value_single, show=False)
                    plt.title(f"SHAP Waterfall Plot - Row {instance_index} - {best_model_name}")
                    plt.tight_layout()
                    plt.savefig(f"artifacts/shap_waterfall_plot_row_{instance_index}.png")
                    plt.close()
                    logging.info(f"SHAP waterfall plot saved to artifacts/shap_waterfall_plot_row_{instance_index}.png")

            except Exception as e:
                logging.warning(f"SHAP explainability skipped: {e}")

            # ✅ Return R² score
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)