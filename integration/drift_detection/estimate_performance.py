import numpy as np
import pandas as pd
import pickle 
import logging
import time
import argparse
import json
import os
import uuid
import copy

import sys
sys.path.append("../..")  # OWAD directory
sys.path.append("../../moudles") # OWAD modules such as calibrator and shifthunter are here
sys.path.append("../../../utils") # the utils directory under the main project

from calibrator import Calibrator
from utils_drift_detection import estimate_precision_OWAD

from utils_general import (
    ipfixTime2unixTS,
    unixTS2ipfixTime,
    whereToLookAtInEachFile,
    build_line_index,
    load_line_index,
    get_flows_by_range
)

from loggers import toFile_timestamped
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

DEFAULT_path2config = "./estimate_performance_config.json"
DEFAULT_log_level = "DEBUG"
LOG_LEVELS = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# DEFAULTS for parameters in config file pertaining to this script
DEFAULT_fitting_evalWindows_count = 8
DEFAULT_fitting_evalWindows_shift = 604800
DEFAULT_fitting_evalWindows_length = 604800
DEFAULT_unseen_evalWindows_count = 8
DEFAULT_unseen_evalWindows_shift = 604800
DEFAULT_unseen_evalWindows_length = 604800
DEFAULT_save_path_parent = "./performance_estimations/"
DEFAULT_metrics = [
    "macro F1",
    "macro F1 common classes", # macro average taken over the classes that are common between predictions and true
    "macro precision",
    "macro precision common classes", # macro average taken over the classes that are common between predictions and true
    "macro recall",
    "macro recall common classes", # macro average taken over the classes that are common between predictions and true
    "accuracy",
    "min F1",
    "min precision",
    "min recall",
    "CM raw counts",
    "CM normalised by row",
    "CM normalised by column"    
]
DEFAULT_epsilon = 0.01
DEFAULT_log_path = "./logs/estimate_performance.log" 
DEFAULT_operates_only_on_unseen_period = False
DEFAULT_reads_unseen_from_IPFIX_records = True
DEFAULT_save_explanations = True
DEFAULT_append_labels_to_explanations = True


# DEFAULT value of parameters related to the modelling config file
# For compatibility, each of the following values should match the corresponding value in build_models.py
DEFAULT_fitting_windows_count = 10
DEFAULT_fitting_windows_start = "2020-07-01 00:00:00.000"
DEFAULT_fitting_windows_length = 4838400 # 8 weeks
DEFAULT_fitting_windows_shift = 9676800  # 2 x 8 weeks
DEFAULT_train_data_portion = 0.7
DEFAULT_train_test_split_random_state = 1
DEFAULT_path_to_models_material_parent = "../classification/models/" # This one should always match (pointing to the same path) the value of DEFAULT_save_path_parent in build_models.py

def main(
    config: dict,
    logger: logging.Logger,
    runID: str,
    selfRunID: str
):
    prefix = f"[run UID = {selfRunID}] - "

    logger.info(prefix + "START ...")
    tic_wholeThing = time.time()
    
    IPFIXPaths = {}
    timestampPaths = {}
    indexPaths = {}
    timestampValuesList = {}
    lineIndex = {}
    
    ### parse performance estimation config parameters
    # only the ones that are not mandatory will be processed here
    fitting_evalWindows_count = config.get("fitting_evalWindows_count", DEFAULT_fitting_evalWindows_count)
    fitting_evalWindows_shift = config.get("fitting_evalWindows_shift", DEFAULT_fitting_evalWindows_shift)
    fitting_evalWindows_length = config.get("fitting_evalWindows_length", DEFAULT_fitting_evalWindows_length)
    unseen_evalWindows_count = config.get("unseen_evalWindows_count", DEFAULT_unseen_evalWindows_count)
    unseen_evalWindows_shift = config.get("unseen_evalWindows_shift", DEFAULT_unseen_evalWindows_shift)
    unseen_evalWindows_length = config.get("unseen_evalWindows_length", DEFAULT_unseen_evalWindows_length)
    path_to_models_material_parent = config.get("path_to_models_material_parent", DEFAULT_path_to_models_material_parent)
    save_path_parent = config.get("save_path_parent", DEFAULT_save_path_parent)
    metrics = config.get("metrics", DEFAULT_metrics)
    log_path = config.get("log_path", DEFAULT_log_path)
    epsilon = config.get("epsilon", DEFAULT_epsilon)
    operates_only_on_unseen_period = config.get("operate_only_on_unseen_period", DEFAULT_operates_only_on_unseen_period)
    reads_unseen_from_IPFIX_records = config.get("read_unseen_from_IPFIX_records", DEFAULT_reads_unseen_from_IPFIX_records)
    save_explanations = config.get("save_explanations", DEFAULT_save_explanations)
    if save_explanations:
        os.makedirs(f"{config['save_path_parent']}/{selfRunID}/explanations", exist_ok = True)

    logger.info(prefix + f"CONFIGS --- modelling reference runID: {runID}, fitting_evalWindows_count: {fitting_evalWindows_count}, fitting_evalWindows_shift: {fitting_evalWindows_shift}, fitting_evalWindows_length: {fitting_evalWindows_length}, unseen_evalWindows_count: {unseen_evalWindows_count}, unseen_evalWindows_shift: {unseen_evalWindows_shift}, unseen_evalWindows_length: {unseen_evalWindows_length}, path_to_models_material_parent: {path_to_models_material_parent}, save_path_parent: {save_path_parent}, metrics: {metrics}, log_path: {log_path}, epsilon: {epsilon}, operates_only_on_unseen_period: {operates_only_on_unseen_period}, reads_unseen_from_IPFIX_records: {reads_unseen_from_IPFIX_records}, save_explanations: {save_explanations}")
    append_labels_to_explanations = config.get("append_labels_to_explanations", DEFAULT_append_labels_to_explanations)


    ### parse modelling config parameters
    with open(f"{path_to_models_material_parent}/{runID}/config.json", "r") as modellingConfigFile:
        modellingConfig = json.load(modellingConfigFile)
        
    n_models = modellingConfig.get("fitting_windows_count", DEFAULT_fitting_windows_count)
    t0_IPFIX = modellingConfig.get("fitting_windows_start", DEFAULT_fitting_windows_start)
    print(f"type(t0_IPFIX): {type(t0_IPFIX)} ------ t0_IPFIX: {t0_IPFIX}")
    t0 = ipfixTime2unixTS(t0_IPFIX)
    fitting_windows_length = modellingConfig.get("fitting_windows_length", DEFAULT_fitting_windows_length)
    fitting_windows_shift = modellingConfig.get("fitting_windows_shift", DEFAULT_fitting_windows_shift)
    train_data_portion = modellingConfig.get("train_data_portion", DEFAULT_train_data_portion)
    train_test_split_random_state = modellingConfig.get("train_test_split_random_state", DEFAULT_train_test_split_random_state)

    if reads_unseen_from_IPFIX_records:
        for fileNameDeviceIdentifier in config["classes"]: 
            IPFIXPaths[fileNameDeviceIdentifier] = []
            timestampPaths[fileNameDeviceIdentifier] = []
            indexPaths[fileNameDeviceIdentifier] = []
            for dirIndex, Dir in enumerate(config["path2LeafIPFIXDirs"]):
                if os.path.exists(f"{Dir}/{fileNameDeviceIdentifier}"):
                    IPFIXPaths[fileNameDeviceIdentifier].append(f"{Dir}/{fileNameDeviceIdentifier}")
                    timestampPaths[fileNameDeviceIdentifier].append(f"{config['path2LeafTimestampDirs'][dirIndex]}/{fileNameDeviceIdentifier.split('.')[0]}.csv")
                    if not os.path.exists(f"{config['path2LeafLineIndexDirs'][dirIndex]}/{fileNameDeviceIdentifier.split('.')[0]}.pkl"):
                        logger.warning(prefix + f"Line index file corresponding to {Dir}/{fileNameDeviceIdentifier} is not there. Building ...")
                        build_line_index(
                            f"{Dir}/{fileNameDeviceIdentifier}",
                            f"{config['path2LeafLineIndexDirs'][dirIndex]}/{fileNameDeviceIdentifier.split('.')[0]}.pkl"
                        )                
                        logger.info(prefix + f"Line index file corresponding to {Dir}/{fileNameDeviceIdentifier} is built ...")
                    indexPaths[fileNameDeviceIdentifier].append(f"{config['path2LeafLineIndexDirs'][dirIndex]}/{fileNameDeviceIdentifier.split('.')[0]}.pkl")
    
            logger.debug(prefix + f"IPFIXPaths populated for {fileNameDeviceIdentifier.split('.')[0]}: IPFIXPaths = {IPFIXPaths[fileNameDeviceIdentifier]}")
            logger.debug(prefix + f"timestampPaths populated for {fileNameDeviceIdentifier.split('.')[0]}: timestampPaths = {timestampPaths[fileNameDeviceIdentifier]}")
            logger.debug(prefix + f"indexPaths populated for {fileNameDeviceIdentifier.split('.')[0]}: indexPaths = {indexPaths[fileNameDeviceIdentifier]}")        
    
            timestampValuesList[fileNameDeviceIdentifier] = [pd.read_csv(path)['timestamp'].values for path in timestampPaths[fileNameDeviceIdentifier]]
            lineIndex[fileNameDeviceIdentifier] = [load_line_index(path) for path in indexPaths[fileNameDeviceIdentifier]]
    
    evals = {}
    
    logger.info(prefix + f"number of different models that will be considered {n_models}")
    for i in range(n_models): 
        evals[f"window_{i}"] = {
            "seen": {},
            "unseen": {}
        }

        # load model (classifier)
        with open(f"{path_to_models_material_parent}/{runID}/models/window_{i}.pkl", "rb") as modelFile:
            model = pickle.load(modelFile)
        logger.info(prefix + f"model index i = {i} | classifier model loaded from {path_to_models_material_parent}/{runID}/models/window_{i}.pkl")
        
        # work out the start and end UNIX timestamps for the corresponding fitting window
        fitting_window_start = t0 + i * fitting_windows_shift
        fitting_window_end = fitting_window_start + fitting_windows_length
        logger.info(prefix + f"model index i = {i} | worked out start <---> end timestamp of the corresponding fitting window: {unixTS2ipfixTime(fitting_window_start)} <---> {unixTS2ipfixTime(fitting_window_end)}")
        
        # load fitting data
        fitting_data_with_TS = pd.read_csv(f"{path_to_models_material_parent}/{runID}/Data/window_{i}.csv")
        selected_columns = [col for col in ["flowEndMilliseconds"] + config["features"] + ["deviceName"] if col in fitting_data_with_TS.columns]
        fitting_data_with_TS = fitting_data_with_TS[selected_columns]
        logger.info(prefix + f"model index i = {i} | loaded corresponding fitting dataset (pre-stored) from {path_to_models_material_parent}/{runID}/Data/window_{i}.csv")

        if not "flowEndMilliseconds" in fitting_data_with_TS.columns:
            if not operates_only_on_unseen_period:
                raise ValueError("operate_only_on_unseen_period is unset (False) while the fitting/seen period dataset lacks timestamps which are necessary for windowing it.")
            else:
                fitting_data_with_TS["flowEndMilliseconds"] = ["2000-01-01 00:00:0.000"] * len(fitting_data_with_TS)  # dummy timestamps for code compatibility        

        fitting_data_with_TS["flowEndMilliseconds"] = fitting_data_with_TS["flowEndMilliseconds"].apply(ipfixTime2unixTS)
        fitting_data_with_TS.dropna(inplace = True)

        if train_data_portion < 1:
            fitting_data_with_TS_train_X, fitting_data_with_TS_test_X, fitting_data_with_TS_train_y, fitting_data_with_TS_test_y = train_test_split(
                    fitting_data_with_TS[["flowEndMilliseconds"] + config["features"]], 
                    fitting_data_with_TS["deviceName"],
                    train_size = train_data_portion, 
                    shuffle = True, 
                    stratify = fitting_data_with_TS["deviceName"],
                    random_state = train_test_split_random_state
                )
            fitting_data_with_TS_test = fitting_data_with_TS_test_X.copy()
            fitting_data_with_TS_test["deviceName"] = fitting_data_with_TS_test_y
            fitting_data_with_TS_test = fitting_data_with_TS_test[fitting_data_with_TS_test["deviceName"].apply(lambda x: x + ".json").isin(config["classes"])] # restrict to a specific list of classes
            logger.debug(prefix + f"model index i = {i} | len(fitting_data_with_TS_test) = {len(fitting_data_with_TS_test)}")
                
        else: # train_data_portion == 1
            logger.warn(prefix + f"train_data_portion is not less than 1, which means all fitting period data is considered as training data.")
            fitting_data_with_TS_train_X, fitting_data_with_TS_train_y = fitting_data_with_TS[["flowEndMilliseconds"] + config["features"]], fitting_data_with_TS["deviceName"]
            
        # construct training-phase confidence scores and calibrators
        fitting_data_with_TS_train = fitting_data_with_TS_train_X.copy()
        fitting_data_with_TS_train["deviceName"] = fitting_data_with_TS_train_y
        fitting_data_with_TS_train = fitting_data_with_TS_train[fitting_data_with_TS_train["deviceName"].apply(lambda x: x + ".json").isin(config["classes"])] # restrict to a specific list of classes
        training_confidence_dict = {}
        calibrator_dict = {}
        predicted_labels = model.predict(fitting_data_with_TS_train[config["features"]])
        for Class in np.unique(fitting_data_with_TS_train["deviceName"]):
            correctly_classified_as_Class = fitting_data_with_TS_train[config["features"]][(predicted_labels == Class) & (fitting_data_with_TS_train["deviceName"] == Class)]
            training_confidence_dict[Class] = model.predict_proba(correctly_classified_as_Class).max(axis = 1)
            cb = Calibrator(correctly_classified_as_Class, method = "Isotonic")
            cb.set_calibrator(training_confidence_dict[Class], is_P_mal=False)
            calibrator_dict[Class] = cb
            print(f"training_confidence_dict and calibrator_dict are updated with items corresponding to class {Class}. Average +/- stdev of training confidence for this class: {np.mean(training_confidence_dict[Class]):.4f} +/- {np.std(training_confidence_dict[Class]):.4f}")
            logger.info(prefix + f"(correct) prediction confidence of training instances + calibrator for class {Class} is constructed")
        print("Done with the training calibrations!")
           
        # metric evaluations for fitting period (seen distribution)
        if not operates_only_on_unseen_period:
            if train_data_portion < 1:
                for ii in range(fitting_evalWindows_count):
                    evalWindowStart = fitting_window_start + ii * fitting_evalWindows_shift
                    evalWindowEnd = min(evalWindowStart + fitting_evalWindows_length, fitting_window_end)
                    evalWindow_test_data = fitting_data_with_TS_test[(fitting_data_with_TS_test["flowEndMilliseconds"] < evalWindowEnd) & (fitting_data_with_TS_test["flowEndMilliseconds"] >= evalWindowStart)]
                    if len(evalWindow_test_data):            
                        evalWindow_test_data_X = evalWindow_test_data[config["features"]]
                        logger.info(prefix + f"model index i = {i} | extracted test X and y data for the (seen/fitting) period {unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)} -- ii = {ii}")
                        evals[f"window_{i}"]["seen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"] = {}
    
                        if any("precision" in metric for metric in metrics):
                            drift_ledger_save_path = f"{config['save_path_parent']}/{selfRunID}/explanations" + f"/window_{i}_seen_{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}.csv"
							precision_per_class, drift_ledger = estimate_precision_OWAD(
								X = evalWindow_test_data_X,
								Y_true = evalWindow_test_data["deviceName"],
								model = model,
								calibrator_dict = calibrator_dict,
								training_confidence_dict = training_confidence_dict,
								save_drift_ledger = save_explanations,
								pvalue_threshold = 0.05,
								drift_ledger_save_path = drift_ledger_save_path
							)

                            logger.info(prefix + f"per-class estimated precisions are calculated for fitting-period evluation window {ii}")
                            # per-class precision
                            if any(metric in ["per-class precision", "precision per-class"] for metric in metrics):
                                evals[f"window_{i}"]["seen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"]["per-class precision"] = precision_per_class
                            # macro precision
                            if any(metric in ["macro precision", "macro-precision"] for metric in metrics):
                                evals[f"window_{i}"]["seen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"]["macro precision"] = np.mean(list(precision_per_class.values()))
                                            
                    else:
                        logger.warn(prefix + f"model index i = {i} | no data instances in (seen/fitting) period {unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)} -- ii = {ii}")
            
            else: # train_data_portion == 1
                logger.warn(prefix + f"train_data_portion is not less than 1, which means all fitting period data is considered as training data. Hence, although operates_only_on_unseen_period is unset (False) no evaluations on the (seen/fitting) period are inevitably being skipped.")
                    

        # metric evaluations for unseen period
        if reads_unseen_from_IPFIX_records:            
            for jj in range(unseen_evalWindows_count):
                evalWindowStart = fitting_window_end + jj * unseen_evalWindows_shift
                evalWindowEnd = evalWindowStart + unseen_evalWindows_length
                records = []
                labels = []
                for fileNameDeviceIdentifier in config["classes"]:
                    IPFIXPaths_thisDev = IPFIXPaths[fileNameDeviceIdentifier]
                    timestampValuesList_thisDev = timestampValuesList[fileNameDeviceIdentifier]
                    lineIndex_thisDev = lineIndex[fileNameDeviceIdentifier]
                    Ranges = whereToLookAtInEachFile(evalWindowStart, evalWindowEnd, timestampValuesList_thisDev)
                    for rr, Range in enumerate(Ranges):
                        if not Range == (0,0):
                            records += get_flows_by_range(IPFIXPaths_thisDev[rr], lineIndex_thisDev[rr], Range[0], Range[1])
                            labels += [fileNameDeviceIdentifier.split(".")[0]] * (Range[1] - Range[0] + 1)            
                evalWindow_data = pd.DataFrame(records)
                evalWindow_data.insert(len(evalWindow_data.columns), "deviceName", labels, False) 
                evalWindow_data.dropna(inplace = True)
                if len(evalWindow_data):
                    evalWindow_data_X = evalWindow_data[config["features"]]
                    evals[f"window_{i}"]["unseen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"] = {}
                    if any("precision" in metric for metric in metrics):
                        drift_ledger_save_path = f"{config['save_path_parent']}/{selfRunID}/explanations" + f"/window_{i}_unseen_{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}.csv"
						precision_per_class, drift_ledger = estimate_precision_OWAD(
							X = evalWindow_data_X,
							Y_true = evalWindow_data["deviceName"],
							model = model,
							calibrator_dict = calibrator_dict,
							training_confidence_dict = training_confidence_dict,
							save_drift_ledger = save_explanations,
							pvalue_threshold = 0.05,
							drift_ledger_save_path = drift_ledger_save_path
						)
                        logger.info(prefix + f"per-class estimated precisions are calculated for unseen evluation window {jj}")
                        # per-class precision
                        if any(metric in ["per-class precision", "precision per-class"] for metric in metrics):
                            evals[f"window_{i}"]["unseen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"]["per-class precision"] = precision_per_class
                        # macro precision
                        if any(metric in ["macro precision", "macro-precision"] for metric in metrics):
                            evals[f"window_{i}"]["unseen"][f"{unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)}"]["macro precision"] = np.mean(list(precision_per_class.values()))
        
                else:
                    logger.warn(prefix + f"model index i = {i} | no data instances in the (unseen) period {unixTS2ipfixTime(evalWindowStart)} <---> {unixTS2ipfixTime(evalWindowEnd)} -- jj = {jj}")
                    
        else:
            # in this case assumptions will be made on the name of the dataset CSV file names specifically: startswith(f"window_{i}_unseen")
            for csv_file in [fileName for fileName in os.listdir(f"{path_to_models_material_parent}/{runID}/Data") if fileName.startswith(f"window_{i}_unseen") and fileName.endswith(".csv")]:
                unseen_data = pd.read_csv(f"{path_to_models_material_parent}/{runID}/Data/{csv_file}")[config["features"] + ["deviceName"]]
                unseen_data_X = unseen_data[config["features"]]
                evals[f"window_{i}"]["unseen"][f"{csv_file.split('.')[0]}"] = {}
                if any("precision" in metric for metric in metrics):
                    drift_ledger_save_path = f"{config['save_path_parent']}/{selfRunID}/explanations" + f"/window_{i}_unseen_{csv_file.split('.')[0]}.csv"
                    
                    precision_per_class, drift_ledger = estimate_precision_OWAD(
						X = unseen_data_X,
						Y_true = unseen_data["deviceName"],
						model = model,
						calibrator_dict = calibrator_dict,
						training_confidence_dict = training_confidence_dict,
						save_drift_ledger = save_explanations,
						pvalue_threshold = 0.05,
						drift_ledger_save_path = drift_ledger_save_path
					)
                    logger.info(prefix + f"per-class estimated precisions are calculated for unseen data {csv_file}")
                    # per-class precision
                    if any(metric in ["per-class precision", "precision per-class"] for metric in metrics):
                        evals[f"window_{i}"]["unseen"][f"{csv_file.split('.')[0]}"]["per-class precision"] = precision_per_class
                    # macro precision
                    if any(metric in ["macro precision", "macro-precision"] for metric in metrics):
                        evals[f"window_{i}"]["unseen"][f"{csv_file.split('.')[0]}"]["macro precision"] = np.mean(list(precision_per_class.values()))
    
    with open(f"{config['save_path_parent']}/{selfRunID}/metrics.json", "w") as metricsFile:
        json.dump(evals, metricsFile)
    
    
    if "plot" in config:
        os.makedirs(f"{config['save_path_parent']}/{selfRunID}/plots", exist_ok = True)
        for groupID in [gID for gID in config["plot"] if not gID.startswith("#")]:
            fig, ax = plt.subplots(figsize = (12, 6))
            # setting x and y limits
            t_min_worked_out = t0
            t_max_worked_out = t0 + (n_models - 1) * fitting_windows_shift + fitting_windows_length + (unseen_evalWindows_count - 1) * unseen_evalWindows_shift + unseen_evalWindows_length
            x_lim = config["plot"][groupID].get("x_lim", (t_min_worked_out, t_max_worked_out))            
            y_lim = config["plot"][groupID].get("y_lim", (0.0, 1))
            ax.set_xlim(tuple(x_lim))
            ax.set_ylim(tuple(y_lim))
            # temporal shades on the seen and unseen period
            for fittingWindowCounter in range(n_models):
                if "seen_period_shades" in config["plot"][groupID]:
                    ax.axvspan(
                        t0 + fittingWindowCounter * fitting_windows_shift, 
                        t0 + fittingWindowCounter * fitting_windows_shift + fitting_windows_length,
                        **config["plot"][groupID]["seen_period_shades"]
                    )
                if "unseen_period_shades" in config["plot"][groupID]:
                    ax.axvspan(
                        t0 + fittingWindowCounter * fitting_windows_shift + fitting_windows_length,
                        t0 + (fittingWindowCounter + 1) * fitting_windows_shift,
                        **config["plot"][groupID]["unseen_period_shades"]
                    )
                    
            # x axis label
            ax.set_xlabel(
                config["plot"][groupID].get("x_label", "Performance metrics"),
                **config["plot"][groupID].get("x_label_font", {"fontsize": 22, "fontweight": "bold"})
            )            
            # y axis label
            ax.set_ylabel(
                config["plot"][groupID].get("y_label", "Performance metrics"),
                **config["plot"][groupID].get("y_label_font", {"fontsize": 22, "fontweight": "bold"})
            )

            # x ticks and x tick labels
            ix = np.linspace(
                start = config["plot"][groupID].get("x_lim", (t_min_worked_out, _))[0], 
                stop = config["plot"][groupID].get("x_lim", (_, t_max_worked_out))[1],
                num = int(np.ceil(
                    (config["plot"][groupID].get("x_lim", (_, t_max_worked_out))[1] - config["plot"][groupID].get("x_lim", (t_min_worked_out, _))[0]) / config["plot"][groupID].get("xtick_step", 1209600)
                )) + 1
            )
            ax.set_xticks(ix)
            ax.set_xticklabels([unixTS2ipfixTime(item) for item in ix], rotation = 45, ha = 'right', rotation_mode = 'anchor')
            
            # y ticks and y tick labels
            yi = np.linspace(
                start = config["plot"][groupID].get("y_lim", (0.0, _))[0], 
                stop = config["plot"][groupID].get("y_lim", (_, 1))[1],
                num = int(np.ceil(
                    (config["plot"][groupID].get("y_lim", (_, 1))[1] - config["plot"][groupID].get("y_lim", (0.0, _))[0]) / config["plot"][groupID].get("ytick_step", 0.05)
                )) + 1
            )
            ax.set_yticks(yi)
            ax.set_yticklabels([f"{val:.4f}" for val in yi])
                        
            # x tick label font
            for xticklabel in ax.get_xticklabels():
                xticklabel.set(**config["plot"][groupID].get("xtick_label_font", {"fontsize": 20, "fontweight": "bold"}))
            # y tick label font
            for yticklabel in ax.get_yticklabels():
                yticklabel.set(**config["plot"][groupID].get("ytick_label_font", {"fontsize": 20, "fontweight": "bold"}))                  
            
            for mi, metric in enumerate(config["plot"][groupID]["metrics"]):
                if not metric in metrics:
                    logger.warn(prefix + f"{metric} is not among the evaluation metrics, the plot request will be ignored.")
                else:
                    xValues = []
                    yValues = []
                    for fittingWindow in evals.keys():
                        for evalWindow in evals[fittingWindow]["seen"]:
                            xValues.append(1/2 * (ipfixTime2unixTS(evalWindow.split("<--->")[0].strip()) + ipfixTime2unixTS(evalWindow.split("<--->")[1].strip())))
                            yValues.append(evals[fittingWindow]["seen"][evalWindow][metric])

                        for evalWindow in evals[fittingWindow]["unseen"]:
                            xValues.append(1/2 * (ipfixTime2unixTS(evalWindow.split("<--->")[0].strip()) + ipfixTime2unixTS(evalWindow.split("<--->")[1].strip())))
                            yValues.append(evals[fittingWindow]["unseen"][evalWindow][metric])

                ax.plot(
                    xValues,
                    yValues,
                    **config["plot"][groupID].get("plot_stroke", [{}] * len(config["plot"][groupID]["metrics"]))[mi]
                )
            ax.legend(**config["plot"][groupID].get("legend", {"loc": "lower right", "framealpha": 0.55, "fontsize": 22}))
            plt.savefig(f"{config['save_path_parent']}/{selfRunID}/plots/{groupID}.pdf", dpi = 300, bbox_inches = "tight")
            plt.savefig(f"{config['save_path_parent']}/{selfRunID}/plots/{groupID}.png", dpi = 300, bbox_inches = "tight")
    

    print("Done!")
    toc_wholeThing = time.time()
    logger.info(prefix + f"END ... time elapsed: {toc_wholeThing - tic_wholeThing}")
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = "Handling input arguments to this script") 

    argparser.add_argument("runID", type = str, help = "Identifier of the modelling run whose results are being evaluated here. This is a required argument, so that the script can locate the underlying data/models.")
    argparser.add_argument("--path2config", type = str, help = "path to the configurations set up file.")
    argparser.add_argument("--log_path", type = str, help = "path to the file to log into.")
    argparser.add_argument("--log_level", type = str, help = "logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    argparser.add_argument("--selfRunID", type = str, help = "Identifier of this run to appear in the logs. If not provided uuid.uuid4() will be used to get a UID.")
    
    args = argparser.parse_args()
    
    runID = args.runID
    
    if not args.selfRunID == None:
        selfRunID = args.selfRunID
    else:
        selfRunID = str(uuid.uuid4())
    
    if not args.path2config == None:
        path2config = args.path2config
    else:
        path2config = DEFAULT_path2config

    with open(path2config, 'r') as confp:
        config = json.load(confp)
        
    if not args.log_path == None:
        log_path = args.log_path
    else:
        log_path = config.get("log_path", DEFAULT_log_path)
        
    if not args.log_level == None:
        log_level = args.log_level
    else:
        log_level = DEFAULT_log_level

    # creating a copy of the config file used for this operation
    os.makedirs(
        f"{config['save_path_parent']}/{selfRunID}",
        exist_ok = True
    )
    os.system(f"cp {path2config} {config['save_path_parent']}/{selfRunID}/config.json")
    
    logger = toFile_timestamped('ESTIMATE PRECISION', log_path, log_level = LOG_LEVELS[log_level])     
    
    main(config, logger, runID, selfRunID)
