from ast import While
from pickle import TRUE
from re import I

import streamlit as st
#EDA packs 
import matplotlib
import pandas as pd
import codecs
#Components packs*
#we called this ver of comp so if we have a new ver our comp will note break
import streamlit.components.v1 as components
import scipy.stats as ss
# Components Pkgs
import streamlit.components.v1 as components
from atom import ATOMClassifier
from atom.data_cleaning import Encoder
import matplotlib.pyplot as plt

# demonstration of the discretization transform
from numpy.random import randn
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

import seaborn as sns

#Machine Learning Algorithmes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import pickle
import joblib


matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({"font.size": 8})
st.set_option("deprecation.showPyplotGlobalUse", False)
def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list





    # st.balloons()
def main():
    #create a page select menu
    st.set_page_config(layout="wide")
    menu=["Explore your data","Machine learning"]
    choice=st.sidebar.selectbox("Menu",menu)
    data_file = st.file_uploader("Upload CSV",type=['csv'])

    preprecess=False
    Nas=list()
    if data_file is not None:
        df = pd.read_csv(data_file)
        i=0
        if(st.checkbox("Header")):
            colnames=[]
            while(i<len(df.columns)):
                name="Col"+str(i)
                colnames.append(name)
                i+=1
           
            df.columns=colnames
            
        if st.checkbox("Select Columns to use"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select Columns", all_columns)
            if(len(selected_columns)>1):
                df = df[selected_columns]
                st.dataframe(df)
        target=st.selectbox(
                    "Select target Column", df.columns.tolist(),
                    index =len(df.columns.tolist())-1
                )
        y=df[target]

        Na = st.text_area("Vlues considered as Na(one value for each line)")

        Nas=Na.splitlines()

            
     
        #st.dataframe(df)
        scale = st.sidebar.checkbox("Scale", False, "scale")
        encode = st.sidebar.checkbox("Encode", False, "encode")
        if(encode):
            maxforleaveoneout=st.sidebar.number_input("maxCatForleaveOneOut",min_value=3,max_value=10)
        impute = st.sidebar.checkbox("Impute", False, "impute")
        if(impute):

              numimpstrat = st.sidebar.selectbox("Imputing strategy for Numerical features ", ["drop","mean","median","most_frequent"])
              catimpstrat = st.sidebar.selectbox("Imputing strategy for Catégorical features ", ["drop","most_frequent"])
        dest = st.sidebar.checkbox("Descritize Target Variable", False, "dest")
        if(dest):
            destBins=st.sidebar.number_input("Choississez le nombre de catégories",min_value=2,max_value=20)
        outliers = st.sidebar.checkbox("Outliers", False, "outliers")
        balancer = st.sidebar.checkbox("Balance", False, "balancer")
  
        placeholder = st.empty()  # Empty to overwrite write statements
        placeholder.write("Data Prepprocessing...")

        # Initialize atom
        df=df.replace(Nas,"?")
        X=df.drop(target,axis=1)
       
        
        atom = ATOMClassifier(X,y, verbose=2, random_state=1)
      

        if impute:
            preprecess=True
          

            placeholder.write("Imputing the missing values...")
            
            atom.impute(strat_num=numimpstrat, strat_cat=catimpstrat,max_nan_cols=0.8)

        if encode:
            preprecess=True
            placeholder.write("Encoding the categorical features...")
            atom.clean(drop_types=None, strip_categorical=False, drop_max_cardinality=True, drop_min_cardinality=False, drop_duplicates=False, drop_missing_target=True, encode_target=True)
            atom.encode(strategy="LeaveOneOut", max_onehot=maxforleaveoneout)
        if(dest):
            if target in categorical_column(df):
                    st.error("Ce target n'est pas continue")
            else:
                data = atom.dataset[target].values.reshape(-1, 1)
                #print(data)
                # discretization transform the raw data
                kbins = KBinsDiscretizer(n_bins=destBins, encode='ordinal', strategy='uniform')
                data_trans = kbins.fit_transform(data)
                df=atom.dataset
                df[target]=data_trans
                #st.write(data_trans[:10, :])
                st.write(plt.hist(data_trans, bins=destBins))
                st.pyplot()
            
        if outliers:
            preprecess=True
            placeholder.write("Pruning values...")
            atom.prune(strategy="EE",include_target=False)


        if scale:
            preprecess=True
            placeholder.write("Scaling the data...")
            atom.scale(strategy="minmax")
        if balancer :
            preprecess=True
            placeholder.write("Balance data values...")
            atom.balance(strategy="Smote")
        placeholder.write("Preprocessing is over...")
        preprecess=False
        df=atom.dataset
        
        st.dataframe(df)
              
        if choice=="Explore your data" :
                df=atom.dataset
                st.title("Automated EDA")
                 

                # Show Columns
                if st.checkbox("Columns Names"):
                    st.write(df.columns)

                # Show Shape
                if st.checkbox("Shape of Dataset"):
                    st.write(df.shape)
                    data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
                    if data_dim == "Columns":
                        st.text("Numbers of Columns")
                        st.write(df.shape[1])
                    elif data_dim == "Rows":
                        st.text("Numbers of Rows")
                        st.write(df.shape[0])
                    else:
                        st.write(df.shape)

                # Select Columns
 

                # Show Value Count
                if st.checkbox("Show Value Counts"):
                    all_columns = df.columns.tolist()
                    selected_columns = st.selectbox("Select Column", all_columns)
                    st.write(df[selected_columns].value_counts())

                # Show Datatypes
                if st.checkbox("Show Data types"):
                    st.text("Data Types")
                    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
                    st.write(df_types.astype(str))

                # Show Summary
                if st.checkbox("Show Summary"):
                    st.text("Summary")
                    st.write(df.describe().T)

                # Plot and visualization
                st.markdown("<h4>Data Visualization</h4>",unsafe_allow_html=True)
                all_columns_names = df.columns.tolist()

                # Correlation Seaborn Plot
                if st.checkbox("Show Correlation Plot"):
                    st.success("Generating Correlation Plot ...")
                    if st.checkbox("Annot the Plot"):
                        st.write(sns.heatmap(df.corr(), annot=True))
                    else:
                        st.write(sns.heatmap(df.corr()))
                    st.pyplot()

                # Count Plot
                all_columns = categorical_column(df)

                if st.checkbox("Show Value Count Plots"):
                    x = st.selectbox("Select Categorical Column", all_columns)
                    st.success("Generating Plot ...")
                    if x:
                        if st.checkbox("Select Second Categorical column"):
                            hue_all_column_name = df[df.columns.difference([x])].columns
                            hue = st.selectbox("Select Column for Count Plot",all_columns )

                            st.write(sns.countplot(x=x, hue=hue, data=df, palette="Set2"))
                        else:
                            st.write(sns.countplot(x=x, data=df, palette="Set2"))
                        st.pyplot()

                # Pie Chart
                if st.checkbox("Show Pie Plot"):
                    selected_columns = st.selectbox("Select Column", all_columns)
                    if selected_columns:
                        st.success("Generating Pie Chart ...")
                        st.write(df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%"))
                        st.pyplot()

                # Customizable Plot
                st.subheader("Customizable Plot")

                type_of_plot = st.selectbox(
                    "Select type of Plot", ["area", "bar", "line", "hist", "box", "kde"]
                )
                selected_columns_names = st.multiselect("Select Columns to plot", all_columns_names)

                if st.button("Generate Plot"):
                    st.success(
                        "Generating Customizable Plot of {} for {}".format(
                            type_of_plot, selected_columns_names
                        )
                    )

                    custom_data = df[selected_columns_names]
                    if type_of_plot == "area":
                        st.area_chart(custom_data)

                    elif type_of_plot == "bar":
                        st.bar_chart(custom_data)

                    elif type_of_plot == "line":
                        st.line_chart(custom_data)

                    elif type_of_plot:
                        custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(custom_plot)
                        st.pyplot()
                #Quantititive-Quantitative Correlation
                if st.checkbox("Qualititive-Qualitative Correlation"):
                    st.subheader("Qualitative Correlation")
                    CATCOLS=categorical_column(df)
                    Cat1=st.selectbox("First Categorical Column",CATCOLS)
                    Cat2=st.selectbox("Second Categorical Column",CATCOLS)
                    x1=df[Cat1]
                    x2=df[Cat2]
                    conf_matrix=pd.crosstab(x1, x2)
                    if conf_matrix.shape[0]==2:
                        correct=False
                    else:
                        correct=True

                    chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]
                    n =sum(conf_matrix.sum())
                    phi2 = chi2/n
                    r,k = conf_matrix.shape
                    #Bias Correction
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
                    rcorr = r - ((r-1)**2)/(n-1)
                    kcorr = k - ((k-1)**2)/(n-1)
                    CRAMER=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
                    st.write(sns.heatmap(conf_matrix, annot=True))
                    st.write("CHi 2 :",chi2)
                    st.write("PHI 2 :",phi2)
                    st.write("Cramer :",CRAMER)
                    st.pyplot()
                if st.checkbox("Quantitative-Quantitative Correlation"):
                    st.subheader("Quantitative Correlation")
                    QuantCOLS=list(set(df.select_dtypes(include=np.number).columns.tolist())-set(categorical_column(df)))
                    Qa1=st.selectbox("First Quantitative Column",QuantCOLS)
                    Qa2=st.selectbox("Second Quantitative Column",QuantCOLS)
                    st.write(sns.regplot(x = Qa1, y = Qa2,  ci = None,scatter_kws={"color": "blue"}, line_kws={"color": "red"}, data = df))
                    st.pyplot()
                  
        elif choice=="Machine learning":
            if preprecess==False:
                st.subheader("Automated Machine Learning")
                st.markdown("> **Select the Model you want to train**")
                modelName = get_model_name()
                st.markdown("> **Adjust the hyperparameters of the selected model**")
                params = get_parameters(modelName)
                model = get_model(modelName,params)
                
                X = df.drop(columns=target)
                y = df[target]
                
                # train test splitting
                st.markdown("### Data spliting")
                help_strat = "Stratification helps getting the same proportion of targets in train and test sets"
                
                test_size = st.slider("Test set size",0.1,0.5)
                strat = st.checkbox(label="Stratification",help=help_strat)
                
                if strat:
                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
                        
                st.write("X_train shape : ",X_train.shape)
                st.write("y_train shape : ",y_train.shape)
                st.write("X_test shape : ",X_test.shape)
                st.write("y_test shape : ",y_test.shape)
                
                # Training the model
                st.markdown("### Model Training and Evaluation")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                #Model evaluaton
                metrics = get_metrics(y_test, y_pred)
                st.write("The Accurracy score : ",metrics["accuracy"])
                st.write("The Precision score : ",metrics["precision"])
                st.write("The Recall score : ",metrics["recall"])
                st.write("The F1 score : ",metrics["f1"])
                
                st.markdown("### Additional evaluation metrics")
                            
                col1, col2 = st.columns(2)
                
                with col1:
                    matrix_plot = st.checkbox("Confusion Matrics")
                with col2:
                    roc_plot = st.checkbox("ROC Curve")
                
                # Confnusion Matrix
                if matrix_plot:
                    cf_matrix = confusion_matrix(y_test,y_pred)
                    plot_confusion_matrix(cf_matrix)
                
                # ROC Curve
                if roc_plot:
                    y_proba = model.predict_proba(X_test)
                    plot_roc_curve(y_test, y_proba)
                    
                # Download 
                st.markdown("### Exporting the Model")
                down = st.checkbox("Export Model",help="Download your trained model")
                if down:
                    col1, col2 = st.columns(2)
                    fpkl = open("filePkl","wb")
                    fjbl = open("fileJbl","wb")
                    pickle.dump(model,fpkl)
                    joblib.dump(model,fjbl)
                    
                    fpkl = open("filePkl","rb")
                    fjbl = open("fileJbl","rb")
                    
                    with col1:
                        st.download_button(label="Download Trained Model Pickle",data=fpkl,file_name="model.pkl")
                    with col2:
                        st.download_button(label="Download Trained Model Joblib",data=fjbl,file_name="model.joblib")
                    
                
                    
            

#returns the choosen model's name
def get_model_name():
    typeOfModel = st.selectbox("Type of Models",("Base Models","Ensemblist Models"))
    if typeOfModel == "Base Models":
        modelName = st.selectbox("Select a Model",("Logistic Regression","SVM","KNN"))
    else:
        modelName = st.selectbox("Select a Model",("Random Forest","Ada Boost"))
    return modelName

#return a list of compatible penalties with given solver
def penalty_select_logReg(solver):
    if solver == "lbfgs":
        penalties = ["none","l2"]
    elif solver == "newton-cg":
        penalties = ["none","l2"]
    elif solver == "liblinear":
        penalties = ["l1","l2"]
    elif solver == "sag":
        penalties = ["none","l2"]
    else:
        penalties = ["none","l1","l2","elasticnet"]
    return penalties

#set the choosen model's parameter ui and return the setted parameters
def get_parameters(modelName):
    params = dict()
    if modelName == "SVM":
        C = st.slider("C value",0.01,10.0)
        kernels = ["rbf", "linear", "poly", "sigmoid", "precomputed"]
        kernel = st.selectbox("Kernel",kernels)
        params["C"] = C
        params["kernel"] = kernel
    elif modelName == "KNN":
        K = st.slider("K neighbors",1,10)
        params["K"] = K
    elif modelName == "Logistic Regression":
        C = st.slider("C value",0.01,10.0)
        solvers = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        solver = st.selectbox("Solver",solvers)
        penalties = penalty_select_logReg(solver)
        penalty = st.selectbox("Penalty",penalties)
        params["C"] = C
        params["solver"] = solver
        params["penalty"] = penalty
    elif modelName == "Random Forest":
        estimators = st.slider("Number of Estimators",1,150)
        max_depth = st.slider("Maximum depth",2,15)
        params["estimators"] = estimators
        params["max_depth"] = max_depth
    else: #modelName == "Ada Boost"
        estimators = st.slider("Number of Estimators",1,150)
        lr = st.slider("Learning rate",0.01,10.0)
        params["estimators"] = estimators
        params["lr"] = lr
    return params
        
#configure and return the choosen machine learning model
def get_model(modelName,params):
    if modelName == "SVM":
        model = SVC(C=params["C"], kernel=params["kernel"])
    elif modelName == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["K"])
    elif modelName == "Logistic Regression":
        model = LogisticRegression(C=params["C"], solver=params["solver"], penalty=params["penalty"])
    elif modelName == "Random Forest":
        model = RandomForestClassifier(n_estimators=params["estimators"], max_depth=params["max_depth"])
    else: #modelName == "Ada Boost"
        model = AdaBoostClassifier(n_estimators=params["estimators"], learning_rate=params["lr"])
    return model

def get_metrics(y_test,y_pred):
    metrics = dict()
    
    metrics["accuracy"] = accuracy_score(y_test,y_pred)
    metrics["precision"] = precision_score(y_test,y_pred)
    metrics["recall"] = recall_score(y_test,y_pred)
    metrics["f1"] = f1_score(y_test,y_pred)
    
    return metrics

def plot_confusion_matrix(cf_matrix):
    fig = plt.figure()
    
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    
    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    # Plot on web app front
    st.pyplot(fig)

def plot_roc_curve(y_test, y_proba):
    # Get the false negative and true positive rates
    fpr, tpr, thresh = roc_curve(y_test, y_proba[:,1], pos_label=1)
    
    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    
    # Compute the AUC Score
    auc_score = roc_auc_score(y_test, y_proba[:,1])
    
    fig = plt.figure()
    # Ploting
    plt.plot(fpr, tpr, linestyle="--", label="AUC : "+str(auc_score), color="red")
    plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
    # Plot details
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # Legend
    plt.legend(loc="best")
    
    # Plot on web app front
    st.pyplot(fig)

if __name__=="__main__":
    main()   
