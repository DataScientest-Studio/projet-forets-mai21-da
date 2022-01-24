import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import random
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split

def render():

# -*- coding: utf-8 -*-


#----------------------------------
# st.text("Importation de packages")
# code='''import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import seaborn as sns
# import streamlit as st
# '''
# st.code(code,language='python')
#----------------
#condition = st.sidebar.selectbox(
 #   "Select the visualization",
  #  ("Int#roduction", "Exploration", "Model Prediction", "Model Evaluation")
#)



#-----------------------
## titre de la page
    st.title("Projet Forêt_modélisation")
    st.write('construire un modèle de prévision du stocks de carbone atmospherique dans la biomasse forestière aérienne et souterraine en fonction de differents à partir de facteurs inlfuençant celci')
    st.markdown('<style>body{background-color:lightblue;}</style>',unsafe_allow_html=True)

    #---------------------------------
    ## lecture de csv
    st.text("Read csv")
    df = pd.read_csv('clean_final_model.csv')

    ##preprocessing-----------------------------------------

    df["net_emission_removals_co2ton"] = df["net_emission_removals_co2ton"]/1000
    df.info()
    #on renommes les colonnes

    dico = {"net_emission_removals_co2ton": "net_carbon_stock_change","EU28_CO2_emissions" : "country_global_emissions"}

    df = df.rename(dico, axis=1)

    code='''df = pd.read_csv('clean_final_model.csv')
    df.head()'''
    st.code(code, language='python')

    #lecture de nouveau dataset----------------------------------
    st.dataframe(data=df)

    #Description des données-------------------------
    code='''df.describe'''
    st.code(code,language='python')
    A=df.describe()
    st.write(A)

    #heatmap------------------------------------
                
    st.subheader("Correlation entre variables")                
    code='''plt.figure(figsize=(16, 16))
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0);
    '''
    plt.figure(figsize=(16, 16))
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0);
    st.code(code, language='python')
    st.write(fig)

    st.caption('Certaines variables sont corrélées entre elles corrélation > 80 et d''autres sont très peu corrélées à la variable net_carbon_stock_change')

    #-Division le dataset en deux jeux de données---------------------------------------


    #la variable cible
    y = df['net_carbon_stock_change']

    #garder le df sans la variable cible
    X=df.drop('net_carbon_stock_change', axis=1, inplace=True)

    #Trier les dates des plus anciennes aux plus récentes avant de créer les valeurs indicatrices.
    X = df.sort_values(by="Year", ascending=True)

    # # Utilisation de la fonction get_dumies sur les variables sous format string en vue de l'étape Standardisation (= création de variables indicatrices)
    X = X.join(pd.get_dummies(X["Year"], prefix="Year"))

    X = X.join(pd.get_dummies(X["Country_Name"], prefix="code"))

    # Suppresion des colonnes au format String

    X = X.drop(["Year", "Country_Name"], axis= 1)

    # Séparation du jeu de données en jeu de d'entrainement et jeu de test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


    ##-----------------------premier model

    from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
    st.subheader("model sur tout le dataset")
    model = LinearRegression()
    model.fit(X,y)
    model.score(X,y)
    code='''model = LinearRegression()
    model.fit(X,y)
    model.score(X,y)'''

    st.code(code, language='python')
    st.write("Score test", model.score(X, y))

    # A voir si on laisse

    # code='''coeffs = list(model.coef_)
    # coeffs.insert(0, model.intercept_)

    # feats = list(X.columns)
    # feats.insert(0, 'intercept')'''

    # pd.DataFrame({'valeur estimée': coeffs}, index=feats)

    #st.dataframe(data=ff)

    # from sklearn.feature_selection import SelectKBest
    # from sklearn.feature_selection import f_regression
    # sk = SelectKBest(f_regression, k=5)
    # sk.fit(X, y)

    # X.columns[sk.get_support()]



    # -----------------------Standardisation du jeu de données
    code='''
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)'''
    st.code(code, language='python')	



    #------------Entrainer differents models
    code='''model_choice=st.selectbox(label="Choix de modèle", options=['Linear Regression', 'Lasso', 'Ridge', 'RandomForestRegressor', 'DecisionTreeRegressor','KNeighborsRegressor'])

    def train_model(model_choice):
        if model_choice=="Linear Regression":
            model = LinearRegression()
        elif model_choice=="Lasso":
            model = Lasso(alpha=1)
        elif model_choice=="Ridge":
            model = RidgeCV(alphas=(0.001,0.01,0.3,0.7,1,10,50,100))
        elif model_choice=="RandomForestRegressor":
            model = RandomForestRegressor(n_jobs=3)
        elif model_choice=="DecisionTreeRegressor":
            model = DecisionTreeRegressor(max_depth=4)
        elif model_choice=="KNeighborsRegressor":
            model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score
    '''

    st.code(code, language='python')

    model_choice=st.selectbox(label="Choix de modèle", options=['Linear Regression', 'Lasso', 'Ridge', 'RandomForestRegressor', 'DecisionTreeRegressor','KNeighborsRegressor'])

    @st.cache
    def train_model(model_choice):
        if model_choice=="Linear Regression":
            model = LinearRegression()
        elif model_choice=="Lasso":
            model = Lasso(alpha=1)
        elif model_choice=="Ridge":
            model = RidgeCV(alphas=(0.001,0.01,0.3,0.7,1,10,50,100))
        elif model_choice=="RandomForestRegressor":
            model = RandomForestRegressor(n_jobs=3)
        elif model_choice=="DecisionTreeRegressor":
            model = DecisionTreeRegressor(max_depth=4)
        elif model_choice=="KNeighborsRegressor":
            model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return score

    #st.write("Score test", model.score(X_test, y_test)) avant def
    st.write("Score test", train_model(model_choice))
    st.caption("Selon les scores, les deux modèles les plus réussis sont la régression linéaire et les crêtes avec des scores de 0,61 et 0,63 respectivement")


    #l'intercept ainsi que les coefficients de chaque variable estimée par le modèle
    model_choice=st.selectbox(label="#Les valeurs ajustées (pred_train)", options=['Linear Regression', 'Lasso', 'Ridge', 'RandomForestRegressor', 'DecisionTreeRegressor','KNeighborsRegressor'])

    def train_model(model_choice):
        if model_choice=="Linear Regression":
            model = LinearRegression()
        elif model_choice=="Lasso":
            model = Lasso(alpha=1)
        elif model_choice=="Ridge":
            model = RidgeCV(alphas=(0.001,0.01,0.3,0.7,1,10,50,100))
        elif model_choice=="RandomForestRegressor":
            model = RandomForestRegressor(n_jobs=3)
        elif model_choice=="DecisionTreeRegressor":
            model = DecisionTreeRegressor(max_depth=4)
        elif model_choice=="KNeighborsRegressor":
            model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        return pred_test
    st.write("pred", train_model(model_choice))


    # #st.write("Score test", model.score(X_test, y_test)) avant def
    #---------------------------

    # coeffs = list(model.coef_)
    # coeffs.insert(0, model.intercept_)

    # feats = list(X.columns)
    # feats.insert(0, 'intercept')

    # pd.DataFrame({'valeur estimée': coeffs}, index=feats)


    # #(d) Afficher le score (R²) du modèle sur l'échantillon d'apprentissage.
    # #(e) Afficher le score obtenu par validation croisée grâce à la fonction cross_val_score().
    # print('Coefficient de détermination du modèle :', model_choice.score(X_train, y_train))
    # print('Coefficient de détermination obtenu par Cv :', cross_val_score(model_choice,X_train,y_train).mean())


    #---------prediction et plot de model
    #the fitted values (pred_train) then the residuals (residuals) of the model.

    code='''#Les valeurs ajustées (pred_train) puis les résidus (residuals) du modèle.
    pred_test = model.predict(X_test)
    plt.scatter(pred_test, y_test)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()));'''
    st.code(code, language='python')

    pred_test = model.predict(X_test)
    plt.scatter(pred_test, y_test)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()));

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pred_test, y_test)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()))
    plt.title("Régression linéaire entre les valeurs prédites et les valeurs réelles")
    plt.xlabel("Valeur réelle")
    plt.ylabel("Valeur prédite")
    plt.legend()

    st.pyplot(fig)


    

    #-------------------
    st.subheader("Nuage de points représentant les résidus en fonction des valeurs de y_train")
    pred_train = model.predict(X_train)
    residus = pred_train - y_train
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_train, residus, color='#980a10', s=15)
    plt.plot((y_train.min(), y_train.max()), (0, 0), lw=3, color='#0a5798');
    st.pyplot(fig)

    st.caption("Les points sont répartis uniformément autour de la droite d'équation  y=0  car la moyenne des résidus avoisine 0.Cependant on constate une structure dans leur répartition. En effet, plus le Net stock change est bas plus les points sont proches de la droite. Plus le net stock change augmente, plus les résidus s'éloignent du centre et deviennent positifs.")

    ##

    residus_norm = (residus-residus.mean())/residus.std()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax=stats.probplot(residus_norm, plot=plt)
    st.pyplot(fig)
    st.caption("un diagramme Quantile-Quantile ou (Q-Q plot) qui permet d'évaluer la pertinence de l'ajustement d'une distribution donnée à un modèle théorique montre que l'hypothèse de normalité est plausible, les points s'alignant approximativement le long de la droite.")
    # plt.show()
    # st.()
    # model_choice=st.selectbox(label="Choix de modèle", options=['Linear Regression', 'Ridge'])
    # @st.cache
    # def train_model(model_choice):
    #     if model_choice=="Linear Regression":
    #         model = LinearRegression()    
    #     elif model_choice=="Ridge":
    #         model = RidgeCV(alphas=(0.001,0.01,0.3,0.7,1,10,50,100))   
    #     pred_test = model.predict(X_test)
    #     return pred_test

    # plt.scatter(pred_test, y_test)
    # plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()));
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot()
    st.title("Conclusion")
    st.caption("Nous avons testé 6 modèles parmi les quels les modèles regression linéar et Ridge fonctionnaient, mais peuvent être encore améliorés. Notamment parce que toutes les variables présentes ont été incluses dans la construction du modèle.Or, des variables ayant peu de relation avec la variable à expliquer, ou un certain nombre de variables comme precipitation et net carbon stock trop corrélées entre elles peuvent entraîner une baisse des résultats")


    st.text('Le score trouvé par le modele est 0.768')
        
        