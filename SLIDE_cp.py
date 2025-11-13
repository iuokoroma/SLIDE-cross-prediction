# Dependencies
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from rpy2.robjects.packages import importr
import rpy2.robjects.pandas2ri as pandas2ri
import rpy2.robjects as robjects
import rpy2
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import itertools
import yaml

def listvec_2_dict(list_vector):
    
    """
    ## listvec_2_dict ##
    
    A recursive function that further unnests an .rds ListVector
    of ListVectors to Python dictionaries with values of type 
    IntVector and StrVector
    
    Args:
        list_vector (ListVector): Nested data in ListVector of ListVectors
        
    Returns:
        dict: Stores the data from R, unnested
        
    """
    if isinstance(list_vector, rpy2.robjects.vectors.ListVector):
        names = list(list_vector.names) if list_vector.names is not \
            None else [None] * len(list_vector)
        vec2dict = {}
        for name, value in zip(names, list_vector):
            key = name if name is not None else f"unnamed_{len(vec2dict)}"
            vec2dict[key] = listvec_2_dict(value)
        return vec2dict
            
    elif isinstance(list_vector, (rpy2.robjects.vectors.IntVector, \
                               rpy2.robjects.vectors.StrVector)):
        vec2lst = list(list_vector)
        return vec2lst if len(vec2lst) > 1 else vec2lst[0]
        
    else:
        return list_vector
        
def path_2_py(path):
    
    """
    ## path_2_py ##
    
    Converts paths from R .rds objects of type ListVector 
    to Python dictionaries
    
    Args:
        path (str): Path to the .rds object
        
    Returns:
        dict: Python object to hold nested structures
        
    """
    # rpy2 method for unwrapping r object in python
    list_vector = robjects.r['readRDS'](path)
    names = list(list_vector.names)
    dict_ = {}

    # breaking down list vectors; each name and extracted
    # element will be a key-value pair
    for name in names:
        obj = list_vector.rx2(name)
        
        if (type(obj) == rpy2.robjects.vectors.IntVector) or \
        (type(obj) == rpy2.robjects.vectors.FloatMatrix) or \
        (type(obj) == rpy2.robjects.vectors.FloatVector):
            dict_[name] = np.array(list_vector.rx2(name))
            if len(dict_[name]) == 1:
                dict_[name] = list_vector.rx2(name)[0]
        elif type(obj) == rpy2.robjects.vectors.ListVector:
            dict_[name] = listvec_2_dict(obj)

    return dict_  

def pred_Z(X, ALF_path):
    
    """
    ## pred_Z ##
    
    Computes latent factor matrix, Z
    
    Args:
        X (pd.DataFrame): X data
        ALF_path (str): Path to AllLatentFactors.rds
        
    Returns: 
        pd.DataFrame: Z_hat matrix
        
    """
    # read path into operable python object
    ALF = path_2_py(ALF_path)

    # read A, C, and Gamma matrices from ALFs
    A_hat = ALF['A']
    C_hat = ALF['C']
    Gamma_hat = ALF['Gamma']

    # if an entry in Gamma_hat is equal to 0, change to a small non-zero number
    Gamma_hat = np.array([1e-10 if i == 0 else i for i in Gamma_hat])

    # computing Z hat matrix
    Gamma_hat_inv = np.diag(Gamma_hat ** (-1))
    G_hat = (A_hat.T @ Gamma_hat_inv @ A_hat) + np.linalg.inv(C_hat)
    Z_hat = np.array(X) @ Gamma_hat_inv @ A_hat @ np.linalg.pinv(G_hat)
    Z_hat = pd.DataFrame(Z_hat)

    return Z_hat

def pairwise_interactions(sigLF_idx, Z):

    """
    ## pairwise_interactions ##
    
    Returns row vector of interaction values between Z terms
    
    Args:
        sigLF_idx (list): List of indices in marginal_vals from SLIDE_res,
                            indicating the significant LFs SLIDE finds 
        Z (pd.DataFrame): Z matrix, contains all LFs SLIDE finds
        
    Returns:
        pd.DataFrame: Results of paired interactions between all latent
                        factors and the significant standalones
        
    """
    # subsetting Z dataframe to contain only columns of significant LFs
    sigLF_idx = [int(i) for i in sigLF_idx]
    sigLF_Z = Z.iloc[:, sigLF_idx]

    # forming interacting terms between the significant LFs and all other LFs
    interaction_terms = [f'{i}.{j}' for i, j in itertools.product(sigLF_Z.columns, Z.columns)]

    # interaction scores are computed by multiplying sigLFs by LFs SLIDE finds
    interactions = np.array([sigLF_Z[i] * Z[j] for i in sigLF_Z.columns for j in Z.columns])
    
    # transposed to be displayed as cells x interacting LFs
    interactions = interactions.T

    # form matrix (df) using interactions data; the terms are column names
    interaction_matrix = pd.DataFrame(interactions, columns=interaction_terms)

    if interaction_matrix.shape == (0, 0):
        flat_df = pd.DataFrame([interaction_matrix.to_numpy().flatten()])
        interaction_matrix = flat_df

    return interaction_matrix

def get_lf_df(SLIDE_LFs_path, Z, interactions=True):

    """
    ## get_lf_df ##
    
    Given access to the latent factor (LF) object and Z matrix, 
    the LF dataframe is created
    
    Args: 
        SLIDE_LFs_path (str): Path to read in 'SLIDE_LFs.rds'
        Z (pd.DataFrame): Z matrix
        interactions (bool): Determines whether you would like to include the interacting LFs
    Returns:
        pd.DataFrame: Returns the LF dataframe, which could include pairwise interactions
        
    """
    # read path into operable python object
    SLIDE_LFs = path_2_py(SLIDE_LFs_path)
    SLIDE_res = SLIDE_LFs['SLIDE_res']

    # significant K number of LFs from SLIDE
    sigK = SLIDE_res['marginal_vars']
    sigK = np.array(sigK)

    # accounts for python indexing starting at 0
    sigK_px = sigK - 1

    # selects significant LF columns
    sig_LF = Z.iloc[:, sigK_px]

    # returns Z matrix of only significant standalone LFs without 
    # the interaction terms (if that is the desired comparison)
    if not interactions:
        return sig_LF

    # significant interaction(s) picked out by SLIDE
    sig_interacts = SLIDE_res['interaction_vars']

    # constructing pairwise interactions
    pw_int = pairwise_interactions(sigK_px, Z)

    # create a standalone and interacting LF matrix if significant interactions are
    # in the pairwise interaction table
    if all(i in pw_int.columns for i in sig_interacts):
        if type(sig_interacts) is int:
            sig_int_data = pw_int.loc[:, [sig_interacts]] # cast as list to perform df.loc
        elif type(sig_interacts) is list:
            sig_int_data = pw_int.loc[:, sig_interacts]
        sig_pw_cols = sig_LF.columns.to_list() + sig_int_data.columns.to_list()
        sig_pw_mtx = pd.DataFrame(np.c_[sig_LF, sig_int_data], columns = sig_pw_cols)
        return sig_pw_mtx

    return sig_LF

def cross_prediction_SLIDE(ALF_path, train_x, train_y, val_x, val_y, SLIDE_LFs_path, interactions=True):
    
    """
    ## cross_prediction_SLIDE ##
    
    Function that cross predicts your train and validation sets
    using statsmodel's GLM (Gaussian)
    
    Args:
        ALF_path (str): Path to AllLatentFactors.rds
        train_x (pd.DataFrame): Training X data
        train_y (pd.DataFrame): Training y data
        val_x (pd.DataFrame): Validation X data
        val_y (pd.DataFrame): Validation y data
        SLIDE_LFs_path (str): Path to SLIDE_LFs.rds
        interactions (bool): Determines whether you would like to include the interacting LFs
        
    Returns:
        dict: SLIDE cross prediction metrics
        
    """
    # creating feature groups
    common_feats = [f for f in train_x.columns if f in val_x.columns]
    missing_feats = [f for f in train_x.columns if f not in val_x.columns]

    # create new column in val_x that includes feature from train_x.
    # it's represented by a 0 vector because that feature is absent in val_x.
    # ensures dimensionality compatibility when running pred_Z
    print("⚠️ Validation set missing features. Adding zeros for missing features.")
    for f in missing_feats:
        val_x[f] = 0
    val_x = val_x[train_x.columns]
    
    # scale data
    scaler = StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)
    val_x = pd.DataFrame(scaler.fit_transform(val_x), columns=val_x.columns)

    print("📊 Generating latent features..." + '\n')
    train_z = pred_Z(train_x, ALF_path)
    val_z = pred_Z(val_x, ALF_path)
    train_lf = get_lf_df(SLIDE_LFs_path, train_z, interactions)
    val_lf = get_lf_df(SLIDE_LFs_path, val_z, interactions)

    # convert dfs to np arrays to fit and predict linear model on SLIDE LFs, uses statsmodel GLM for linear model
    train_lf_const = sm.add_constant(np.array(train_lf))
    val_lf_const = sm.add_constant(np.array(val_lf))
    ols_model = sm.GLM(np.array(train_y), train_lf_const, family=sm.families.Gaussian()).fit()
    tpred_y = ols_model.predict(train_lf_const)
    vpred_y = ols_model.predict(val_lf_const)

    # compute AUC from sklearn.metrics
    train_auc = roc_auc_score(train_y, tpred_y)
    val_auc = roc_auc_score(val_y, vpred_y)

    # compute ROC from sklearn.metrics
    roc_score_train = roc_curve(train_y, tpred_y)
    roc_score_val = roc_curve(val_y, vpred_y)

    # cross prediction metrics output as a dictionary
    cp_metrics = {'roc_score_train': roc_score_train,
              'roc_score_val': roc_score_val,
              'train_auc': round(train_auc, 2),
              'val_auc': round(val_auc, 2),
              'model': ols_model,
              'lfs': train_lf.columns.to_list()}
    
    return cp_metrics

def show_cross_prediction(yaml_path, val_x_path, val_y_path, interactions=True, save_plot_path=None):

    """
    ## show_cross_prediction ##
    
    Shows cross prediction plot and returns metrics/model information
    
    Args:
        yaml_path (str): Path to yaml_params.yaml
        val_x_path (str): Path to val_x.csv
        val_y_path (str): Path to val_y.csv
        interactions (bool): Determines whether you would like to include the interacting LFs
        save_plot_path (str): Will save ROC plot to a predefined path if one isn't defined

    Shows:
        plot: Training and Validation AUC curves
        
    Returns:
        dict: SLIDE cross prediction metrics
        
    """
    # loading yaml file as a dict to safely access that data
    with open(yaml_path, 'r') as file:
        input_ = yaml.safe_load(file)

    # accessing the path to additional access AllLatentFactors and SLIDE_LFs
    out_path = input_['out_path']
    ALF_path = out_path + "AllLatentFactors.rds"
    SLIDE_LFs_path = out_path + "SLIDE_LFs.rds"

    print("Loading training and validation data")
    train_x = pd.read_csv(input_['x_path'], index_col=0)
    train_y = pd.read_csv(input_['y_path'], index_col=0)
    val_x = pd.read_csv(val_x_path, index_col=0)
    val_y = pd.read_csv(val_y_path, index_col=0)

    # running SLIDE cross prediction function
    cp_SLIDE = cross_prediction_SLIDE(ALF_path, train_x, train_y, val_x, val_y, \
                                SLIDE_LFs_path, interactions)

    # breaking up ROC score tuples to generate plots
    fpr_train, tpr_train, thresh_train = cp_SLIDE['roc_score_train']
    fpr_val, tpr_val, thresh_val = cp_SLIDE['roc_score_val']

    # generate ROC plot
    plt.figure(figsize=(6,6))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='-', lw=0.75)
    plt.plot(fpr_train, tpr_train, label=f"Train AUC = {cp_SLIDE['train_auc']:.3f}")
    plt.plot(fpr_val, tpr_val, label=f"Val AUC = {cp_SLIDE['val_auc']:.3f}")
    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve")
    plt.legend()

    # if there isn't already a plot path, create a plot and save that
    if save_plot_path is None:
        plt.savefig("roc_plot.jpg")
        plt.show()
    else:
        plt.savefig(save_plot_path)
        plt.show()
    
    return cp_SLIDE
