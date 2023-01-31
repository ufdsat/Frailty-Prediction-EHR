from cgi import test
import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar


ccs_dict = {"Anemia" : ["ccs_" + str(i) for i in range(59, 62)],
            "Cardiac dysrhythmias" : ["ccs_106", "ccs_107"],
            "Cancer" : ["ccs_" + str(i) for i in range(11, 22)] + 
                       ["ccs_" + str(i) for i in range(24, 37)] + 
                       ["ccs_41"],
            "Cerebrovascular disease" : ["ccs_" + str(i) for i in range(109, 114)],
            "Coornary atherosclerosis" : ["ccs_101"], 
            "Diabetes" : ["ccs_49", "ccs_50"],
            "CHF" : ["ccs_108"], 
            "Hypertension" : ["ccs_98", "ccs_99"],
            "Kidney Disease" : ["ccs_" + str(i) for i in range(156, 159)] + ["ccs_161"],
            "Liver Disease" : ["ccs_151"],
            "Lung Disease" : ["ccs_127", "ccs_128"],
            "Thyroid Disorder" : ["ccs_48"],
            "Osteoporosis" : ["ccs_206"], 
            "Heart Valve Disease" : ["ccs_96"], 
            "Syncope" : ["ccs_245"], 
            "Myocardial Infarction" : ["ccs_100"],
            "Melanoma of Skin" : ["ccs_22"], 
            "Peptic ulcer" : ["ccs_139", "ccs_140", "ccs_144", "ccs_153"],
            "Chronic skin ulcer" : ["ccs_199"],
            "Urinary System Disease" : ["ccs_159", "ccs_160", "ccs_162", "ccs_163", "ccs_164"],
            "Atrial Disease" : ["ccs_114", "ccs_115", "ccs_117"],
            "Arthritis" : ["ccs_202", "ccs_203"],
            "Fall" : ["ccs_2603"],
            "Malaise and fatigue" : ["ccs_252"],
            "Slow Gait Speed" : ["Slow_Walking_Speed_Value"],
            "Parkinson Disease" : ["ccs_79"],
            "Back problem" : ["ccs_205"],
            "Eye Disease" : ["ccs_"+str(i) for i in range(86, 92)],
            "Ear Disorder" : ["ccs_94"],
            "Congnitive Disorder" : ["ccs_653"],
            "Anxiety" : ["ccs_651"],
            "Weight Loss" : ["Unintended_Weight_Loss_Value"],
            "Dizziness" : ["ccs_93"],
            "Nutritional deficiencies" : ["ccs_52"]
            }

feature_dict = {"age" : "Age", 
                    "HEMOGLOBIN" : "Hemoglobin",
                    "Chronic obstructive pulmonary disease and bronchiectasis" : "Chronic obstructive pulmonary disease and bronchiectasis",
                    "Edu_Years" : "Years of education",
                    "HEMATOCRIT" : "Hematocrit", 
                    "ASA_Anest_Record" : "American Society of Anesthesiologists score",
                    'Other nutritional; endocrine; and metabolic disorders' : 'Other nutritional, endocrine, and metabolic disorders', 
                    'sex' : 'Gender',
                    'Disorders of lipid metabolism' : 'Disorders of lipid metabolism', 
                    'Mood disorders' : 'Mood disorders',
                    'Diabetes mellitus with complications' : 'Diabetes mellitus with complications',
                    'Other nervous system disorders' : 'Other nervous system disorders', 
                    'Patient_Type' : 'Visit type',
                    'Other connective tissue disease' : 'Other connective tissue disease', 
                    'Osteoarthritis' : 'Osteoarthritis',
                    'Spondylosis; intervertebral disc disorders; other back problems' : 'Spondylosis, intervertebral disc disorders, and other back problems',
                    'Congestive heart failure; nonhypertensive' : 'Congestive heart failure (nonhypertensive)',
                    'Marital_Status' : 'Marital status',
                    'Hypertension with complications and secondary hypertension' : 'Hypertension with complications and secondary hypertension',
                    'complication_sum' : 'Complication sum',
                    'Other circulatory disease' : 'Other circulatory disease',
                    'Diabetes mellitus without complication' : 'Diabetes mellitus without complication', 
                    'PLATELET_COUNT' : 'Platelet count',
                    'Other and unspecified benign neoplasm' : 'Other and unspecified benign neoplasm',
                    'Screening and history of mental health and substance abuse codes' : 'History of mental health and substance abuse',
                    'Charlson_Comorbidity_Index' : 'Charlson comorbidity index', 
                    'Anxiety disorders' : 'Anxiety disorders',
                    'Esophageal disorders' : 'Esophageal disorders', 
                    'Other aftercare' : 'Other aftercare',
                    'Coronary atherosclerosis and other heart disease' : 'Coronary atherosclerosis and other heart disease',
                    'Genitourinary symptoms and ill-defined conditions' : 'Genitourinary symptoms and ill-defined conditions',
                    'Other gastrointestinal disorders' : 'Other gastrointestinal disorders', 
                    'Essential hypertension' : 'Essential hypertension',
                    'Thyroid disorders' : 'Thyroid disorders', 
                    'EmployeeStatus' : 'Employement status', 
                    'Allergic reactions' : 'Allergic reactions',
                    'Residual codes; unclassified' : 'Residual CCS codes (unclassified)', 
                    'Race' : 'Race', 
                    'Chronic kidney disease' : 'Chronic kidney disease',
                    'Cardiac dysrhythmias' : 'Cardiac dysrhythmias',
                    'Other screening for suspected conditions (not mental disorders or infectious disease)' : 'Other screening for suspected conditions (not mental disorders or infectious disease)',
                    'Diverticulosis and diverticulitis' : 'Diverticulosis and diverticulitis',
                    'Hemorrhoids' : 'Hemorrhoids',
                    'Abdominal hernia' : 'Abdominal hernia',
                    'Other disorders of stomach and duodenum' : 'Other disorders of stomach and duodenum',
                    'Gastritis and duodenitis' : 'Gastritis and duodenitis',
                    'Deficiency and other anemia' : 'Deficiency and other anemia',
                    'Other diseases of kidney and ureters' : 'Other diseases of kidney and ureters',
                    'Cancer of bladder' : 'Cancer of bladder',
                    'Hyperplasia of prostate' : 'Hyperplasia of prostate',
                    'Cancer of prostate' : 'Cancer of prostate',
                    'Calculus of urinary tract' : 'Calculus of urinary tract',
                    'Other diseases of bladder and urethra' : 'Other diseases of bladder and urethra',
                    'Cancer of kidney and renal pelvis' :  'Cancer of kidney and renal pelvis',
                    'Other hereditary and degenerative nervous system conditions' : 'Other hereditary and degenerative nervous system conditions',
                    'Parkinson`s disease' : 'Parkinson`s disease',
                    'Other acquired deformities' : 'Other acquired deformities',
                    'Osteoporosis' : 'Osteoporosis',
                    'Complication of device; implant or graft' : 'Complication of device; implant or graft',
                    'E Codes: Adverse effects of medical care' : 'Adverse effects of medical care',
                    'Other non-epithelial cancer of skin' : 'Other non-epithelial cancer of skin',
                    'Other bone disease and musculoskeletal deformities' : 'Other bone disease and musculoskeletal deformities',
                    'Other upper respiratory disease' : 'Other upper respiratory disease',
                    'Other ear and sense organ disorders' : 'Other ear and sense organ disorders',
                    'Cancer of head and neck' : 'Cancer of head and neck',
                    'Other upper respiratory infections' : 'Other upper respiratory infections',
                    'Acute posthemorrhagic anemia' : 'Acute posthemorrhagic anemia',
                    'Secondary malignancies' : 'Secondary malignancies',
                    'Nutritional deficiencies' : 'Nutritional deficiencies',
                    'Fluid and electrolyte disorders' : 'Fluid and electrolyte disorders',
                    'Substance-related disorders' : 'Substance-related disorders',
                    'Other lower respiratory disease' : 'Other lower respiratory disease',
                    'Peripheral and visceral atherosclerosis' : 'Peripheral and visceral atherosclerosis',
                    'Aortic; peripheral; and visceral artery aneurysms' : 'Aortic; peripheral; and visceral artery aneurysms', 
                    'Heart valve disorders' : 'Heart valve disorders',
                    'Complications of surgical procedures or medical care' : 'Complications of surgical procedures or medical care',
                    'Occlusion or stenosis of precerebral arteries' : 'Occlusion or stenosis of precerebral arteries',
                    'Conduction disorders' : 'Conduction disorders',
                    'Cataract' : 'Cataract',
                    'Other eye disorders' : 'Other eye disorders',
                    'Cancer of breast' : 'Cancer of breast',
                    'Medical examination/evaluation' : 'Medical examination/evaluation',
                    'Coagulation and hemorrhagic disorders' : 'Coagulation and hemorrhagic disorders',
                    'Acute and unspecified renal failure' : 'Acute and unspecified renal failure',
                    'Pleurisy; pneumothorax; pulmonary collapse' : 'Pleurisy; pneumothorax; pulmonary collapse',
                    'Cancer of bronchus; lung' : 'Cancer of bronchus; lung',
                    'Pulmonary heart disease' : 'Pulmonary heart disease',
                    'Blindness and vision defects' : 'Blindness and vision defects',
                    'Glaucoma' : 'Glaucoma',
                    'Retinal detachments; defects; vascular occlusion; and retinopathy' : 'Retinal detachments',
}               

def cal_draw_shap(service):
    '''
    For each service
        1. draw the shap value on test set
        2. save the rank of the feature importance for the heat map
    '''
    
    if service == "ALL":
        result = "/result/result-XGBoost"
    else:
        result = "/result/result-" + service
    
    shap_testfold = [i for i in os.listdir("./../" + result + "/SHAP Values/") if "test" in i]
    data_testfold = [i for i in os.listdir("./../" + result + "/Splited Data/") if "test" in i]
    
    testshap = []
    testdata = []

    for fold in shap_testfold:
        shap_value = pd.read_pickle("./../" + result + "/SHAP Values/"+fold)
        testshap.append(shap_value)
    
    testshap1 = pd.concat(testshap, axis=0).sort_index()


    for fold in data_testfold:
        tdata = pd.read_pickle("./../" + result + "/Splited Data/"+fold)
        testdata.append(tdata)
    
    testdata1 = pd.concat(testdata, axis=0).sort_index()
    testdata1.columns = [feature_dict[col] for col in  testdata1.columns]
    print(testdata1.columns)
    shap.summary_plot(testshap1.to_numpy(), testdata1, show=False, max_display=15)
    
    plt.savefig("./../" + result + "/shap_plot"+service+".png")
    
    # generate rank of feature importance
    
    vals = np.abs(testshap1.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(testshap1.columns, vals)),
                                  columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)
    shap_importance["ranking"] = range(1, shap_importance.shape[0]+1)
    shap_importance.to_csv("./../" + result + "/feature_ranking/feature_ranking_"+service+".csv", index=False)
    
    
def calculate_rockwood_index(Service):
    pat_data = pd.read_csv("./../data/pat_data_new.csv")
    ccs_data = pd.read_csv("./../data/ccs_data_new.csv")
    frailty_components_data = pd.read_csv("./../data/weak_grip_strength_corrected.csv")
    #'mmse_3word_repeat', 'mmse_3word_recall'
    pat_data.drop(['mmse_3word_repeat', 'mmse_3word_recall'], axis=1, inplace=True)

    Services = np.unique(pat_data.Service.dropna())
    if Service == "ALL":
        pass
    else:
        pat_data = pat_data.loc[pat_data.Service == Service, ]

    pat_data = pat_data.merge(ccs_data, on="studyid", how="left")
    pat_data = pat_data.merge(frailty_components_data[["studyid", "Slow_Walking_Speed_Value", "Unintended_Weight_Loss_Value"]],
                              on="studyid", how="left")

    
    for key, value in ccs_dict.items():
        pat_data[key] = np.where(pat_data[value].sum(axis=1) > 0, 1, 0) 
    
    deficits = [i for i in ccs_dict.keys()]
    pat_data["deficit_count"] = np.sum(pat_data[deficits], axis=1)
    rock_wood_data = pat_data[['studyid', "deficit_count", "Frailty"]]
    rock_wood_data["rock_wood_index"] = np.round(rock_wood_data.deficit_count/34, 2)

    min_rockwood_index = rock_wood_data.rock_wood_index.min()
    max_rockwood_index = rock_wood_data.rock_wood_index.max()
    median_rockwood_index = rock_wood_data.rock_wood_index.median()
    mean_rockwood_index = rock_wood_data.rock_wood_index.mean()

    y = rock_wood_data.Frailty
    pred = rock_wood_data.rock_wood_index
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    auc_score = auc(fpr, tpr)   

    rock_wood_desc = {'min_rockwood_index' : min_rockwood_index, 
                      'max_rockwood_index' : max_rockwood_index,
                      'median_rockwood_index' : median_rockwood_index,
                      'mean_rockwood_index' : mean_rockwood_index,
                      'auc' : auc_score}

    myData = []
    for i in np.arange(np.min(rock_wood_data.rock_wood_index), np.max(rock_wood_data.rock_wood_index), 0.01):
        y_pred = np.where(rock_wood_data.rock_wood_index > i, 1, 0)
    
        accuracy = accuracy_score(rock_wood_data.Frailty, y_pred)
        f1 = f1_score(rock_wood_data.Frailty, y_pred)
        precision = precision_score(rock_wood_data.Frailty, y_pred)
        sensitivity = recall_score(rock_wood_data.Frailty, y_pred)
        tn, fp, fn, tp = confusion_matrix(rock_wood_data.Frailty, y_pred).ravel()
        specificity = tn / (tn+fp)
    
        myData.append([i, accuracy, f1, precision, sensitivity, specificity])
    
    cut_off_data = pd.DataFrame(myData, columns=["Cut-off", "Accuracy", "F1-socre", "Precision", "Sensitivity", "Specificity"])

    path = "./../result/Rockwood Index/" + Service + "/"
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    cut_off_data.to_csv(path + Service + "_cut_off_data.csv", index=False)
    rock_wood_data.to_csv(path + Service + "_rockwood_data.csv", index=False)
    return rock_wood_data, rock_wood_desc, cut_off_data

def calculate_rockwood_index_severity(Service):
    pat_data = pd.read_csv("./../data/pat_data_severity.csv")
    ccs_data = pd.read_csv("./../data/ccs_data_new.csv")
    frailty_components_data = pd.read_csv("./../data/weak_grip_strength_corrected.csv")
    #'mmse_3word_repeat', 'mmse_3word_recall'
    pat_data.drop(['mmse_3word_repeat', 'mmse_3word_recall'], axis=1, inplace=True)

    Services = np.unique(pat_data.Service.dropna())
    if Service == "ALL":
        pass
    else:
        pat_data = pat_data.loc[pat_data.severity == Service, ]

    pat_data = pat_data.merge(ccs_data, on="studyid", how="left")
    pat_data = pat_data.merge(frailty_components_data[["studyid", "Slow_Walking_Speed_Value", "Unintended_Weight_Loss_Value"]],
                              on="studyid", how="left")

    
    for key, value in ccs_dict.items():
        pat_data[key] = np.where(pat_data[value].sum(axis=1) > 0, 1, 0) 
    
    deficits = [i for i in ccs_dict.keys()]
    pat_data["deficit_count"] = np.sum(pat_data[deficits], axis=1)
    rock_wood_data = pat_data[['studyid', "deficit_count", "Frailty"]]
    rock_wood_data["rock_wood_index"] = np.round(rock_wood_data.deficit_count/34, 2)

    min_rockwood_index = rock_wood_data.rock_wood_index.min()
    max_rockwood_index = rock_wood_data.rock_wood_index.max()
    median_rockwood_index = rock_wood_data.rock_wood_index.median()
    mean_rockwood_index = rock_wood_data.rock_wood_index.mean()

    y = rock_wood_data.Frailty
    pred = rock_wood_data.rock_wood_index
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    auc_score = auc(fpr, tpr)   

    rock_wood_desc = {'min_rockwood_index' : min_rockwood_index, 
                      'max_rockwood_index' : max_rockwood_index,
                      'median_rockwood_index' : median_rockwood_index,
                      'mean_rockwood_index' : mean_rockwood_index,
                      'auc' : auc_score}

    myData = []
    for i in np.arange(np.min(rock_wood_data.rock_wood_index), np.max(rock_wood_data.rock_wood_index), 0.01):
        y_pred = np.where(rock_wood_data.rock_wood_index > i, 1, 0)
    
        accuracy = accuracy_score(rock_wood_data.Frailty, y_pred)
        f1 = f1_score(rock_wood_data.Frailty, y_pred)
        precision = precision_score(rock_wood_data.Frailty, y_pred)
        sensitivity = recall_score(rock_wood_data.Frailty, y_pred)
        tn, fp, fn, tp = confusion_matrix(rock_wood_data.Frailty, y_pred).ravel()
        specificity = tn / (tn+fp)
    
        myData.append([i, accuracy, f1, precision, sensitivity, specificity])
    
    cut_off_data = pd.DataFrame(myData, columns=["Cut-off", "Accuracy", "F1-socre", "Precision", "Sensitivity", "Specificity"])

    path = "./../result/Rockwood Index/" + Service + "/"
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    cut_off_data.to_csv(path + Service + "_cut_off_data.csv", index=False)
    rock_wood_data.to_csv(path + Service + "_rockwood_data.csv", index=False)
    return rock_wood_data, rock_wood_desc, cut_off_data

def calculate_rockwood_index_severity_level(Service):
    pat_data = pd.read_csv("./../data/pat_data_severity_level.csv")
    ccs_data = pd.read_csv("./../data/ccs_data_new.csv")
    frailty_components_data = pd.read_csv("./../data/weak_grip_strength_corrected.csv")
    #'mmse_3word_repeat', 'mmse_3word_recall'
    pat_data.drop(['mmse_3word_repeat', 'mmse_3word_recall'], axis=1, inplace=True)

    Services = np.unique(pat_data.Service.dropna())
    if Service == "ALL":
        pass
    else:
        pat_data = pat_data.loc[pat_data.Level == Service, ]

    pat_data = pat_data.merge(ccs_data, on="studyid", how="left")
    pat_data = pat_data.merge(frailty_components_data[["studyid", "Slow_Walking_Speed_Value", "Unintended_Weight_Loss_Value"]],
                              on="studyid", how="left")

    
    for key, value in ccs_dict.items():
        pat_data[key] = np.where(pat_data[value].sum(axis=1) > 0, 1, 0) 
    
    deficits = [i for i in ccs_dict.keys()]
    pat_data["deficit_count"] = np.sum(pat_data[deficits], axis=1)
    rock_wood_data = pat_data[['studyid', "deficit_count", "Frailty"]]
    rock_wood_data["rock_wood_index"] = np.round(rock_wood_data.deficit_count/34, 2)

    min_rockwood_index = rock_wood_data.rock_wood_index.min()
    max_rockwood_index = rock_wood_data.rock_wood_index.max()
    median_rockwood_index = rock_wood_data.rock_wood_index.median()
    mean_rockwood_index = rock_wood_data.rock_wood_index.mean()

    y = rock_wood_data.Frailty
    pred = rock_wood_data.rock_wood_index
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    auc_score = auc(fpr, tpr)   

    rock_wood_desc = {'min_rockwood_index' : min_rockwood_index, 
                      'max_rockwood_index' : max_rockwood_index,
                      'median_rockwood_index' : median_rockwood_index,
                      'mean_rockwood_index' : mean_rockwood_index,
                      'auc' : auc_score}

    myData = []
    for i in np.arange(np.min(rock_wood_data.rock_wood_index), np.max(rock_wood_data.rock_wood_index), 0.01):
        y_pred = np.where(rock_wood_data.rock_wood_index > i, 1, 0)
    
        accuracy = accuracy_score(rock_wood_data.Frailty, y_pred)
        f1 = f1_score(rock_wood_data.Frailty, y_pred)
        precision = precision_score(rock_wood_data.Frailty, y_pred)
        sensitivity = recall_score(rock_wood_data.Frailty, y_pred)
        tn, fp, fn, tp = confusion_matrix(rock_wood_data.Frailty, y_pred).ravel()
        specificity = tn / (tn+fp)
    
        myData.append([i, accuracy, f1, precision, sensitivity, specificity])
    
    cut_off_data = pd.DataFrame(myData, columns=["Cut-off", "Accuracy", "F1-socre", "Precision", "Sensitivity", "Specificity"])

    path = "./../result/Rockwood Index/" + Service + "/"
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    cut_off_data.to_csv(path + Service + "_cut_off_data.csv", index=False)
    rock_wood_data.to_csv(path + Service + "_rockwood_data.csv", index=False)
    return rock_wood_data, rock_wood_desc, cut_off_data

def compare_xgboost_frailty_index(Service, cut_off):
    '''
    Calculate the prediction of frailty using frailty index for the given cut_off point.
    Compare that with the prediction from XGBoost
    
    Return:
        1. Frequency table
        2. Number of samples that XGBoost provided the correct prediction but frailty index didn't
        3. Number of samples that XGBoost provided the wrong prediction but frailty index provided the correct one
        4. Test results from Mcnemar's test
    '''
    index_path = "./../result/Rockwood Index/" + Service + "/"
    frailty_index_data = pd.read_csv(index_path + Service + "_rockwood_data.csv")
    frailty_index_data['pred'] = np.where(frailty_index_data["rock_wood_index"] > cut_off, 1, 0)
    
    if Service == "ALL":
        Service = "XGBoost"
        
    test_fold = [i for i in os.listdir("./../result/result-" + Service + "/Predict Result/") if "test" in i]
    prediction_xgboost = []
    for fold in test_fold:
        pred = pd.read_pickle("./../result/result-" + Service + "/Predict Result/" + fold)
        prediction_xgboost.append(pred)
        
    prediction_xgboost = pd.concat(prediction_xgboost).sort_index()
    
    XGBoost_correct = prediction_xgboost == frailty_index_data.Frailty
    RW_correct = frailty_index_data['pred'] == frailty_index_data.Frailty
    
    matrix = pd.crosstab(XGBoost_correct, RW_correct)
    
    result = mcnemar(matrix, exact=True)
    
    print("resulted matrix: ")
    print(matrix)
    print("Number of samples that XBGoost provided the correct prediction but frailty index didn't", matrix.values[1, 0])
    print("Number of samples that frailty index provided the correct prediction but XGBoost didn't", matrix.values[0, 1])
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
    

def cal_draw_shap(service):
    '''
    For each service
        1. draw the shap value on test set
        2. save the rank of the feature importance for the heat map
    '''
    
    if service == "ALL":
        result = "/result/result-XGBoost"
    else:
        result = "/result/result-" + service
    
    shap_testfold = [i for i in os.listdir("./../" + result + "/SHAP Values/") if "test" in i]
    data_testfold = [i for i in os.listdir("./../" + result + "/Splited Data/") if "test" in i]
    
    testshap = []
    testdata = []

    for fold in shap_testfold:
        shap_value = pd.read_pickle("./../" + result + "/SHAP Values/"+fold)
        testshap.append(shap_value)
    
    testshap1 = pd.concat(testshap, axis=0).sort_index()


    for fold in data_testfold:
        tdata = pd.read_pickle("./../" + result + "/Splited Data/"+fold)
        testdata.append(tdata)
    
    testdata1 = pd.concat(testdata, axis=0).sort_index()
    testdata1.columns = [feature_dict[col] for col in  testdata1.columns]
    shap.summary_plot(testshap1.to_numpy(), testdata1, show=False, max_display=15)
    
    plt.savefig("./../" + result + "/shap_plot"+service+".png")
    
    # generate rank of feature importance
    
    vals = np.abs(testshap1.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(testshap1.columns, vals)),
                                  columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                               ascending=False, inplace=True)
    shap_importance["ranking"] = range(1, shap_importance.shape[0]+1)
    shap_importance.to_csv("./../" + result + "/feature_ranking_"+service+".csv", index=False)
