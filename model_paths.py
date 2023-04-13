import warnings
import joblib

if 'google.colab' in str(get_ipython()):
    model_dir = "BayesianOptimiserMgAlloy/models"
else:
    model_dir = "models"



warnings.filterwarnings('ignore')

models = {"gp_UTS_normalized": joblib.load(f"{model_dir}/gp_UTS_normalized"),
          "gp_Ductility_normalized": joblib.load(f"{model_dir}/gp_Ductility_normalized"),
          "gp_UTS": joblib.load(f"{model_dir}/gp_UTS"),
          "gp_Ductility": joblib.load(f"{model_dir}/gp_Ductility"),
          "rf_UTS": joblib.load(f"{model_dir}/rf_UTS"),
          "rf_Ductility": joblib.load(f"{model_dir}/rf_Ductility")
          }
