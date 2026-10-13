
!pip install optuna rdkit

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time


import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


from sklearn import svm, metrics, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, accuracy_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import pickle
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
import os
import ast
import numpy as np
import pathlib

"""# Functions"""

def smiles_to_fp(smiles, method="", n_bits=2048):

    # converting smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Warning: Invalid SMILES string: {smiles}. Returning None.")
        return np.zeros(n_bits)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=int)
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol), dtype=int)
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol), dtype=int)
    else:

        print(f"Warning: Wrong method specified: {method}. Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=int)

def ml_performance(ml_model, x_test, y_test, verbose=True):

    test_prob = ml_model.predict_proba(x_test)[:,1]


    test_pred = ml_model.predict(x_test)

    accuracy = accuracy_score(y_test, test_pred)
    sens = recall_score(y_test, test_pred)
    spec = recall_score(y_test, test_pred, pos_label=0)
    auc = roc_auc_score(y_test, test_prob)

    if verbose:
         print("performance results")
         print(f"Accuracy: {accuracy:.2}")
         print(f"Sensitivity: {sens:.2f}")
         print(f"Specificity: {spec:.2f}")
         print(f"AUC: {auc:.2f}")

    return accuracy, sens, spec, auc

def calculate_enrichment_factor(label_true, label_score, percentage):


  num_total_actives = np.sum(label_true)
  num_total_molecules = len(label_true)

  sorted_indices = np.argsort(label_score, kind='stable')[::-1]
  label_true_sorted = label_true[sorted_indices]

  num_top_molecules = int((percentage / 100) * num_total_molecules)


  num_top_actives = np.sum(label_true_sorted[:num_top_molecules])

  ef = (num_top_actives / num_top_molecules) / (num_total_actives / num_total_molecules)

  return ef

def model_training_and_validation(ml_model, name, splits, verbose=True):
    x_train, x_test, y_train, y_test = splits
    ml_model.fit(x_train, y_train)
    accuracy, sens, spec, auc = ml_performance(ml_model, x_test, y_test, verbose)

    test_prob = ml_model.predict_proba(x_test)[:,1]

    ef_0_5_percent = calculate_enrichment_factor(y_test, test_prob, 0.5)
    ef_1_percent = calculate_enrichment_factor(y_test, test_prob, 1)
    ef_2_percent = calculate_enrichment_factor(y_test, test_prob, 2)




    return accuracy, sens, spec, auc, ef_0_5_percent, ef_1_percent, ef_2_percent

"""### Root Directory"""

data_dir = Path(".")

"""# Pre-Processing"""

import pathlib

dude_dir = pathlib.Path(data_dir, "DUD-E")
pre_processed_dir = pathlib.Path(data_dir, "pre-processed")
pre_processed_dir.mkdir(parents=True, exist_ok=True)

for target in tqdm(dude_dir.iterdir()):
  if "ampc" in target.name:
        continue

  target_dir = pathlib.Path(target)

  for ism_file in target_dir.glob("*.ism"):
    if ism_file.name == "actives_final.ism":
      df_actives = pd.read_csv(ism_file, sep='\s+', header=None, names=['smiles','dude_id', 'chembl_id'], usecols=['smiles'])
      df_actives['is_active'] = 1.0
    elif ism_file.name == "decoys_final.ism":
      df_decoys = pd.read_csv(ism_file, sep='\s+', header=None, names=['smiles','dude_id'], usecols=['smiles'])
      df_decoys['is_active'] = 0.0
    else:
      print("error")

  df_merged = pd.concat([df_actives, df_decoys], ignore_index=True)

  df_merged["fp"] = df_merged["smiles"].apply(lambda x: smiles_to_fp(x, method="morgan3"))

  df_merged = df_merged.sample(frac=1) #added this to shuffle them

  output_file_name = f"{target.name}.csv"
  output_file_path = Path(pre_processed_dir, output_file_name)
  print(output_file_path)
  df_merged.to_csv(output_file_path, sep="\t", index=False)

  output_feather_file_path = Path(pre_processed_dir, target.name + ".feather")
  print(output_feather_file_path)
  df_merged.to_feather(output_feather_file_path)

"""# Models

## Cross-Validated Model Loop
"""

from sklearn.model_selection import StratifiedKFold
pre_processed_dir = Path(data_dir, "pre-processed") #using path to create path to directory
models_directory = "models"
Path(data_dir, models_directory).mkdir(parents=True, exist_ok=True)


classifiers = {
  "RF": RandomForestClassifier(n_estimators= 10,   criterion="entropy", random_state=42),
  "SVM": SVC(kernel='linear', random_state=42, probability=True),
  "KNN": KNeighborsClassifier(n_neighbors=5), #can add "random_state=42" if data is suffled to control randomness
  "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
}

results = []
for protein_file in tqdm(pre_processed_dir.glob("*.feather"), desc="Processing proteins"):  #  glob used to find files
    protein_name = protein_file.stem
    print(protein_name)
    df_training = pd.read_feather(protein_file)
    X = np.vstack(df_training.fp.values)
    y = df_training.is_active.values

    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

    results_dict = {
        'Protein': protein_name,
    }

    for classifier_name, classifier in classifiers.items():
        # to sttore results for each fold
        fold_accuracies = []
        fold_sensitivities = []
        fold_specificities = []
        fold_aucs = []
        fold_ef_0_5_percents = []
        fold_ef_1_percents = []
        fold_ef_2_percents = []




        for train_index, test_index in kf.split(X,y):

            train_fold_fp = X[train_index]
            train_fold_label = y[train_index]

            test_fold_fp = X[test_index]
            test_fold_label = y[test_index]

            splits = [train_fold_fp, test_fold_fp, train_fold_label, test_fold_label]

            accuracy, sens, spec, auc, ef_0_5_percent, ef_1_percent, ef_2_percent = model_training_and_validation(classifier, classifier_name, splits, verbose=False)

            fold_accuracies.append(accuracy)
            fold_sensitivities.append(sens)
            fold_specificities.append(spec)
            fold_aucs.append(auc)
            fold_ef_0_5_percents.append(ef_0_5_percent)
            fold_ef_1_percents.append(ef_1_percent)
            fold_ef_2_percents.append(ef_2_percent)




        # average results across folds
        results_dict[f"{classifier_name}_Accuracy"] = np.mean(fold_accuracies)
        results_dict[f"{classifier_name}_Sensitivity"] = np.mean(fold_sensitivities)
        results_dict[f"{classifier_name}_Specificity"] = np.mean(fold_specificities)
        results_dict[f"{classifier_name}_AUC"] = np.mean(fold_aucs)
        results_dict[f"{classifier_name}_EF_0.5%"] = np.mean(fold_ef_0_5_percents)
        results_dict[f"{classifier_name}_EF_1%"] = np.mean(fold_ef_1_percents)
        results_dict[f"{classifier_name}_EF_2%"] = np.mean(fold_ef_2_percents)



        # save model
        model_filename = f"{protein_name}_{classifier_name}_model.pkl"
        model_file_path = Path(data_dir, models_directory, model_filename)

        with open(model_file_path, 'wb') as f:
          pickle.dump(classifier, f) #using pickle

    # savings protein results
    results.append(results_dict)


# save all results to file
df_results = pd.DataFrame.from_dict(results)
results_directory = "results"
results_path_dir = Path(data_dir, results_directory)
results_path_dir.mkdir(parents=True, exist_ok=True)
df_results.to_csv(results_path_dir / "all_results_stratkfold_all.csv", index=False)

"""# Sorting Results"""

df_results = pd.read_csv(Path(data_dir, "results", "all_results_stratkfold_all.csv"), sep=",")
df_results.sort_values(by='Protein', ascending=True)
df_results.head()

"""# Plotting Results

## ROC Curve
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm


models_directory = "models"
models_path_dir = Path(data_dir, models_directory)
results_directory = "results"
results_path_dir = Path(data_dir, results_directory)
results_path_dir.mkdir(parents=True, exist_ok=True)
pdf_path = Path(results_path_dir, "ROCs_Graphed.pdf")
pre_processed_dir = pathlib.Path(data_dir, "pre-processed")
pre_processed_dir.mkdir(parents=True, exist_ok=True)
df_results = pd.read_csv(Path(data_dir, "results", "all_results_stratkfold_all.csv"), sep=",") #change name to your results file

def plot_roc_curves(classifiers, protein_name, X, y, pdf):

    # split so the ROC is measured on data the model has not been trained on
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))

    for classifier_name, classifier in classifiers.items():

        model = clone(classifier)
        model.fit(x_train, y_train)

        test_prob = model.predict_proba(x_test)[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_test, test_prob)
        auc = roc_auc_score(y_test, test_prob)

        ax.plot(fpr, tpr, label=f"{classifier_name} (AUC = {auc:.2f})")

    ax.set_title(f"ROC Curves for {protein_name}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.legend(loc="lower right")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    pdf.savefig(fig)
    plt.close(fig)

with PdfPages(pdf_path) as pdf:
    for protein_file in tqdm(pre_processed_dir.glob("*.feather"), desc="Generating ROC curves"):
        protein_name = protein_file.stem

        df_training = pd.read_feather(protein_file)
        X = np.vstack(df_training.fp.values)
        y = df_training.is_active.values

        plot_roc_curves(classifiers, protein_name, X, y, pdf)

        print(f"ROC curve generated for {protein_name}")

print(f"ROC curves PDF generated at: {pdf_path}")

"""## Plotting AUCs"""

df_results = pd.read_csv(Path(data_dir, "results", "all_results_stratkfold_all.csv"), sep=",") #change file to your results file
df_results.head()

import datetime

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

y_increments = np.arange(0.825, 1.0, 0.025)


auc_scores_rf = df_results['RF_AUC'].tolist()
auc_scores_svm = df_results['SVM_AUC'].tolist()
auc_scores_knn = df_results['KNN_AUC'].tolist()
auc_scores_mlp = df_results['MLP_AUC'].tolist()
protein_names = df_results['Protein'].tolist()

results_directory = "results"
results_path_dir = Path(data_dir, results_directory)
results_path_dir.mkdir(parents=True, exist_ok=True)
pdf_path = Path(results_path_dir,  "Results_Graphed.pdf")

#dont have latex installed so i set it to false

with PdfPages(pdf_path) as pdf:
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(protein_names, auc_scores_rf, marker='o', linestyle='-')
    plt.xlabel("Proteins")
    plt.ylabel("AUC Scores")
    plt.title("AUC Scores for Different Proteins RF")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.yticks(y_increments)
    plt.savefig(results_path_dir / 'auc_scores_rf.png')
    pdf.savefig()
    plt.close()

    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(protein_names, auc_scores_svm, marker='o', linestyle='-')
    plt.xlabel("Proteins")
    plt.ylabel("AUC Scores")
    plt.title("AUC Scores for Different Proteins SVM")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.yticks(y_increments)
    plt.savefig(results_path_dir / 'auc_scores_svm.png')
    pdf.savefig()
    plt.close()

    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(protein_names, auc_scores_knn, marker='o', linestyle='-')
    plt.xlabel("Proteins")
    plt.ylabel("AUC Scores")
    plt.title("AUC Scores for Different Proteins KNN")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.yticks(y_increments)
    plt.savefig(results_path_dir / 'auc_scores_knn.png')
    pdf.savefig()
    plt.close()

    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(protein_names, auc_scores_mlp, marker='o', linestyle='-')
    plt.xlabel("Proteins")
    plt.ylabel("AUC Scores")
    plt.title("AUC Scores for Different Proteins MLP")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.yticks(y_increments)
    plt.savefig(results_path_dir / 'auc_scores_mlp.png')
    pdf.savefig()
    plt.close()



    #putting my info on file
    d = pdf.infodict()
    d['Title'] = 'Plotted Results'
    d['Author'] = 'Cian Loughlin'
    d['Subject'] = 'Predicting activities of compounds based on SMILES'
    d['Keywords'] = 'Random Forest SVM KNN MLP DUD-E SMILES'
    d['CreationDate'] = datetime.datetime(2025, 4, 3)
    d['ModDate'] = datetime.datetime.today()
#close outside loop
pdf.close()

"""# Inference

## Start
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

new_prepro_directory = "New_Data_Pre_Processed"
new_prepro_path_dir = Path(data_dir, new_prepro_directory)
new_prepro_path_dir.mkdir(parents=True, exist_ok=True)
dude_dir = pathlib.Path(data_dir, "DUD-E")


df_actives = None
exp_data_directory = "experimental-data"
exp_data_path_dir = Path(data_dir, exp_data_directory)
exp_data_path_dir.mkdir(parents=True, exist_ok=True)
csv_file_name = None



for csv_file in exp_data_path_dir.glob("*.csv"):
  if csv_file.name == "unseen_actives_v3.csv":
    df_actives = pd.read_csv(csv_file, sep='\t', header=0, names=['Smiles'])
    csv_file_name = csv_file.stem
    break

if df_actives is not None:
  df_actives['Smiles'] = df_actives['Smiles'].astype(str)
  df_actives["fp"] =  df_actives["Smiles"].apply(lambda x: smiles_to_fp(x, method="morgan3"))

else:
  print(f"Warning: 'unseen_actives.csv' not found in {exp_data_directory}")

if csv_file_name is not None:
    output_feather_file_name = f"{csv_file_name}_preprocessed.feather"
    output_feather_file_path = Path(data_dir, new_prepro_directory, output_feather_file_name)
    print(output_feather_file_path)
    df_actives.to_feather(output_feather_file_path)
else:
    print("Error: 'unseen_actives_v3.csv' file not found. Cannot proceed.")

from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

models_directory = "models"
models_path_dir = Path(data_dir, models_directory)


all_predictions = {}

targets=[]
classifiers = []

for model_file in models_path_dir.glob("*.pkl"):
  target, classifier_name, _ = model_file.stem.split("_")
  if target not in targets:
    targets.append(target)

  if classifier_name not in classifiers:
    classifiers.append(classifier_name)

print(targets)
print(classifiers)

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

models_directory = "models"
models_path_dir = Path(data_dir, models_directory)
scoring_directory = "scoring"
scoring_path_dir = Path(data_dir, scoring_directory)
scoring_path_dir.mkdir(parents=True, exist_ok=True)


all_predictions = {}

for processed_file in tqdm(new_prepro_path_dir.glob("*.feather"), desc="Processing proteins"):  # using glob to get files
    pro_file_name = processed_file.stem
    if processed_file.name == "unseen_actives_v3_preprocessed.feather":   #can change this to whatever file you want to use
      df_data = pd.read_feather(processed_file)
      X = np.vstack(df_data["fp"].values)
      smiles_data = df_data['Smiles'].tolist()

      print(f"Processing: {pro_file_name}")

      for target in targets:
        target_preds = {

        }

        for classifier in classifiers:
          model_file = models_path_dir / f"{target}_{classifier}_model.pkl"
          with open(model_file, 'rb') as f:
            model = pickle.load(f)

          preds = model.predict(X)
          target_preds[classifier] = preds

        # majority vote across the classifiers, counting votes so ties do not break the shape
        votes = pd.DataFrame(target_preds).sum(axis=1)
        consensus_prediction = (votes > len(classifiers) / 2).astype(int).tolist()

        for i, smiles in enumerate(smiles_data):
            all_predictions.setdefault(smiles, {})[target] = consensus_prediction[i]




results = pd.DataFrame.from_dict(all_predictions, orient='index')
results.index.name = 'Smiles'
results.reset_index(inplace=True)
results = results[['Smiles'] + targets]
results.to_csv(scoring_path_dir / "predictions_3.csv", index=False)

"""# Hyperparamter Tuning"""

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def mlp_objective(trial, x_train_val, x_val, y_train_val, y_val):

    mlp_clf_params = {
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-3, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True)
    }

    n_layers = trial.suggest_int("n_layers", 10, 100)
    layers =[]

    for i in range(n_layers):
      layers.append(trial.suggest_int(f"n_units_l{i}", 10, 300))

    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    mlp_clf = MLPClassifier(**mlp_clf_params, hidden_layer_sizes=tuple(layers), activation=activation)
    n_max_iter = 400

    classes = np.unique(y_train_val)

    for step in range(n_max_iter):
      if step == 0:
          mlp_clf.partial_fit(x_train_val, y_train_val, classes=classes)
      else:
          mlp_clf.partial_fit(x_train_val, y_train_val)

      intermediate_pred = mlp_clf.predict_proba(x_val)
      intermediate_value = np.sqrt(mean_squared_error(y_val, intermediate_pred[:, 1]))

      trial.report(intermediate_value, step)

      if trial.should_prune():
        raise optuna.TrialPruned()

    p_val = mlp_clf.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, p_val))

    return rmse

def optuna_mlp_study(x_train, y_train):

    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.2)

    study = optuna.create_study(
        direction = "minimize",
        study_name = "Optuna MLPClassifier Study",
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    )

    study.optimize(
        lambda trial: mlp_objective(
          trial,
          x_train_val=x_train_val,
          y_train_val=y_train_val,
          x_val=x_val,
          y_val=y_val,
        ),
        n_trials=100,
        n_jobs=10
    )

    best_params = study.best_trial.params
    print(f"Best trial: score {study.best_trial.value}, params {best_params}")

    return best_params

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


pre_processed_dir = pathlib.Path(data_dir, "pre-processed")


protein_params = []

for protein_file in tqdm(pre_processed_dir.glob("*.feather"), desc="Processing proteins"):
    protein_name = protein_file.stem
    print(f"Processing: {protein_name}")


    df_training = pd.read_feather(protein_file)
    X = np.vstack(df_training.fp.values)
    y = df_training.is_active.tolist()


    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    best_params = optuna_mlp_study(x_train, y_train)

    params_dict = {
        'Protein': protein_name,
        'Best_Parameters': best_params
    }

    protein_params.append(params_dict)

    print(f"Best parameters for {protein_name}: {best_params}")
    break


# save all results to file
df_params = pd.DataFrame.from_dict(protein_params)
param_dir = "Optimised-Parameters"
param_dir_path = Path(data_dir, param_dir)
param_dir_path.mkdir(parents=True, exist_ok=True)
df_params.to_csv(param_dir_path / "Optimised__MLP_Parameters.csv", index=False)

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def svm_objective(trial, x_train_val, x_val, y_train_val, y_val):

    svm_clf_params = {
        'C': trial.suggest_float("svc_c", 1e-10, 1e10, log=True),
        'gamma': trial.suggest_categorical("svc_gamma", ["auto", "scale"]),
        'kernel': trial.suggest_categorical("svc_kernel", ["linear", "poly", "rbf", "sigmoid"])
    }

    svm_clf = SVC(**svm_clf_params, probability=True, random_state=42)
    svm_clf.fit(x_train_val, y_train_val)

    p_val = svm_clf.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, p_val))

    return rmse

def optuna_svm_study(x_train, y_train):

    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.2)

    study = optuna.create_study(
        direction='minimize',
        study_name ="Optuna SVMClassifier Study"
    )

    study.optimize(
        lambda trial: svm_objective(
          trial,
          x_train_val=x_train_val,
          y_train_val=y_train_val,
          x_val=x_val,
          y_val=y_val,
        ),
        n_trials=100,
        n_jobs=10
    )

    best_params = study.best_trial.params
    print(f"Best trial: score {study.best_trial.value}, params {best_params}")

    return best_params

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


pre_processed_dir = pathlib.Path(data_dir, "pre-processed")


protein_params = []

for protein_file in tqdm(pre_processed_dir.glob("*.feather"), desc="Processing proteins"):
    protein_name = protein_file.stem
    print(f"Processing: {protein_name}")


    df_training = pd.read_feather(protein_file)
    X = np.vstack(df_training.fp.values)
    y = df_training.is_active.tolist()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    best_params = optuna_svm_study(x_train, y_train)

    params_dict = {
        'Protein': protein_name,
        'Best_Parameters': best_params
    }

    protein_params.append(params_dict)

    print(f"Best parameters for {protein_name}: {best_params}")
    break


df_params = pd.DataFrame.from_dict(protein_params)
param_dir = "Optimised-Parameters"
param_dir_path = Path(data_dir, param_dir)
param_dir_path.mkdir(parents=True, exist_ok=True)
df_params.to_csv(param_dir_path / "Optimised_Parameters_SVM.csv", index=False)

def knn_objective(trial, x_train_val, x_val, y_train_val, y_val):

    knn_clf_params = {
        'n_neighbors': trial.suggest_int("knn_n_neighbours", 1, 20),
        'weights': trial.suggest_categorical("knn_weight", ["uniform", "distance"]),
        'algorithm': trial.suggest_categorical("knn_algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    }

    knn_clf = KNeighborsClassifier(**knn_clf_params)
    knn_clf.fit(x_train_val, y_train_val)

    p_val = knn_clf.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, p_val))

    return rmse

def optuna_knn_study(x_train, y_train):

    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.2)

    study = optuna.create_study(
        direction="minimize",
        study_name ="Optuna KNNClassifier Study"
    )

    study.optimize(
        lambda trial: knn_objective(
          trial,
          x_train_val=x_train_val,
          y_train_val=y_train_val,
          x_val=x_val,
          y_val=y_val,
        ),
        n_trials=100,
        n_jobs=10
    )

    best_params = study.best_trial.params
    print(f"Best trial: score {study.best_trial.value}, params {best_params}")

    return best_params

def rf_objective(trial, x_train_val, x_val, y_train_val, y_val):

    rf_clf_params = {
        'n_estimators': trial.suggest_int("rf_n_estimators", 5, 150, log=True),
        'criterion': trial.suggest_categorical("rf_criterion", ["gini", "entropy", "log_loss"])
    }

    rf_clf = RandomForestClassifier(**rf_clf_params, random_state=42)
    rf_clf.fit(x_train_val, y_train_val)

    p_val = rf_clf.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, p_val))

    return rmse

def optuna_rf_study(x_train, y_train):

    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=0.2)

    study = optuna.create_study(
        direction="minimize",
        study_name ="Optuna RFClassifier Study"
    )

    study.optimize(
        lambda trial: rf_objective(
          trial,
          x_train_val=x_train_val,
          y_train_val=y_train_val,
          x_val=x_val,
          y_val=y_val,
        ),
        n_trials=100,
        n_jobs=10
    )

    best_params = study.best_trial.params
    print(f"Best trial: score {study.best_trial.value}, params {best_params}")

    return best_params
