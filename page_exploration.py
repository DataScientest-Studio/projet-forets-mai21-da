from ssl import Options
from turtle import color
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


def fusion_EmissionsCO2Aviation_plot(df):
    df_agg = df.groupby(['Year']).sum().reset_index()
    
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(df_agg['Year'], df_agg['CO2_aviation_sum'], color='r')
    plt.xticks(rotation=90)
    return fig


#def Variation_temperature_plot(df1):

    #fig = plt.figure(figsize=(10, 10))
    #px.bar(df1,x="Code_country",y="Valeur")
    #plt.xticks(rotation=90)
    #return fig


#def Precipitations_plot(df2):
        
    #fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df2['Year'], df2['Value'], color='r')
    #plt.xticks(rotation=90)
    #return fig


#def Surface_Foret_plot(df3):
        
    #fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df3['year'], df3['surface_pct'], color='r')
    #plt.xticks(rotation=90)
    #return fig


#def Surface_Agricole_plot(df4):
        
    #fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df4['Year'], df4['value_TA_perc'], color='r')
    #plt.xticks(rotation=90)
    #return fig


#def Target_plot(df5):
        
    ##fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df5['Year'], df5['net_emission_removals_co2ton'], color='r')
    #plt.xticks(rotation=90)
    #return fig


#def Emissions_CO2_Aviation_plot(df6):
        
    #fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df6['Year'], df6['CO2_million_sum'], color='r')
    #plt.xticks(rotation=90)
    #return fig


#def Emissions_CO2_Globales_plot(df7):
        
    #fig = plt.figure(figsize=(10, 10))
    #plt.scatter(df7['Year'], df7['EU28_CO2_emissions'], color='r')
    #plt.xticks(rotation=90)
    #return fig



def render():
    

    

    #image = Image.open('foret.jpg')
    #st.image(image)
    
    st.write("##")
   
    st.write ("Nous avons décidé de nous concentrer sur la période 1990-2020 afin de trouver des datasets suffisamment larges, et couvrant au minimum 10 années entre toutes les données disponibles. ll nous a fallu prendre en compte la nécessité d’inclure des données annexes pour entrer certaines données de notre modélisation (impact de nouveaux projets législatifs européens - Green Deal, croissance économique, état des forêts, captation de CO2 moyenne ainsi que des données sur l’aviation et les émissions de CO2 en U.E).")
    st.write ("Il est également important de noter que la zone géographique que nous avons prise en compte est L'Union-Européenne des 27 pays + le Royaume-Uni. Ce dernier étant un acteur majeur sur la périodde ciblée, il nous était impossible de le mettre de côté.")
    st.write ("##")
    st.write ("##")

    #Dataset 1 sur la variation de température

    df1 = pd.read_csv('clean_temp_variation.csv')

    st.subheader("1) Données de variations de température")
    st.write("##")

    dictionnaire = {'Valeur': 'variation_temp_deg'}
    df1 = df1.rename(dictionnaire, axis=1)

    year_options_1 = df1["Annee"].unique().tolist()
    year_1 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_1, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020], key="1")
    country_options_1 = df1["Country_Name"].unique().tolist()
    country_1 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_1,["Austria","Belgium","Bulgaria","Croatia","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovak Republic","Slovenia","Spain","Sweden","United Kingdom"], key="11")

    df1 = df1[df1["Country_Name"].isin(country_1)]
    df1 = df1[df1["Annee"].isin(year_1)]
    st.write("##")
    st.dataframe(df1.head(10))
    st.write("Évolution des variations de températures (source FAO 2021 – données de 1990 à 2020)")
    

    fig1 = px.box(df1,x="Annee",y="variation_temp_deg")

    fig1_1 = px.line(df1,x="Annee",y="variation_temp_deg", color="Code_country")
    st.write("##")
    
    st.plotly_chart(fig1_1)

    st.write ("Premier constat réalisé sur le dataset contenant la variation de température moyenne de surface par année et par pays : nous avons une hausse de cette variation entre 1990 et 2020. À noter également que nous avons deux diminutions importantes de la variation en 1996 et en 2010 (les variations de température ont été normalisées).")

    st.plotly_chart(fig1)
       
    st.write ("On observe également une augmentation généralisée des températures observées pendant la période et ce sur l’intégralité des pays inclus dans le dataset (voir boxplot ci-dessus). On peut compléter l’analyse faite à partir de ce graphique en montrant que l’augmentation de la température moyenne accélère au fur et à mesure des années. On observe en effet dans le graphique de droite une augmentation de l’écart par rapport à la température moyenne observée qui va en augmentant entre 1990 et 2020, ainsi qu’une chute de cette variation durant les années 1996 et 2010.")

    st.write("##")
    st.write("##")



    #Dataset 2 sur les précipitations

    df2 = pd.read_csv('clean_precipitation.csv')

    dictionnaire = {'Value': 'precipitation_m3'}
    df2 = df2.rename(dictionnaire, axis=1)
    
    st.subheader("2) Données de précipitations")  
    st.write("##")
    year_options_2 = df2["Year"].unique().tolist()
    year_2 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_2, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015], key="2")
    country_options_2 = df2["Country_or_Area"].unique().tolist()
    country_2 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_2,["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland","France","Germany","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","United Kingdom of Great Britain and Northern Ireland"],key="22")

    df2 = df2[df2["Country_or_Area"].isin(country_2)]
    df2 = df2[df2["Year"].isin(year_2)]
    st.write("##")
    st.dataframe(df2.head(10))
    st.write("Évolution des précipitations année par année (source ONU 2021 – données 1990 à 2016)")
    fig2 = px.box(df2,x="Country_Code",y="precipitation_m3", color="Country_Code")

    fig2_1 = px.line(df2,x="Year",y="precipitation_m3", color="Country_Code")
        
        
    st.plotly_chart(fig2)
    st.write ("On constate une disparité assez forte de la pluviométrie en fonction de la taille mais aussi de l’emplacement géographique des pays sur le continent européen. Cette disparité des précipitations en fonction de la géographie se confirme dans le graphique suivant. Si on constate une diminution relative des précipitations au fil des années malgré quelques années à forte pluviométrie dues à des évènements météorologiques exceptionnels (tempêtes entraînant des précipitations soudaines très intenses), cela n’empêche pas l’augmentation ou la diminution des précipitations dans certains pays (Belgique augmentation des précipitations et Espagne baisse).")

    st.plotly_chart(fig2_1)

    st.write("##")
    st.write("##")




    #Dataset 3 sur la surface des forêtes en % de la surface du territoire

    df3 = pd.read_csv('clean_surface_foret.csv')
        
    st.subheader("3) Données de surface forestière en pourcentage ")
    st.write("##")
    year_options_3 = df3["year"].unique().tolist()
    year_3 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_3, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018], key="3")
    country_options_3 = df3["Country_Name"].unique().tolist()
    country_3 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_3,["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovak Republic","Slovenia","Spain","Sweden","United Kingdom"])

    df3 = df3[df3["Country_Name"].isin(country_3)]
    df3 = df3[df3["year"].isin(year_3)]
    st.write("##")
    st.dataframe(df3.head(10))
    st.write("Évolution de la surface forestière en pourcentage (source Banque Mondiale 2021 – données de 1990 à 2018)")
    fig3 = px.box(df3,x="year",y="surface_pct")
    fig3_1 = px.line(df3,x="year",y="surface_pct", color="Country_Code")

    st.plotly_chart(fig3_1)
    st.write ("On constate une augmentation relative de la surface forestière pour de nombreux pays entre 1990 et 2018 (par ex. de 45 à 47% du territoire couvert par les forêts pour l’Autriche ou de 30 à 35% sur la période pour la Bulgarie. Il s’agira ensuite de déterminer si cela se confirme par une capacité accrue d’absorption des émissions de CO2.")

    st.plotly_chart(fig3)
    

    st.write ("Une comparaison des années 1990, 1998 et 2018 permet de valider cette analyse. On constate donc bien une augmentation de la surface du territoire couverte par des forêts dans la majorité des pays inclus dans le dataset (U.E et Royaume Uni).")

    st.write("##")
    st.write("##")




    #Dataset 4 sur la surface des agricole en % de la surface du territoire

    df4 = pd.read_csv('clean_agriculture_perc.csv')

    dictionnaire = {'value_TA_perc': 'terres_agricoles_perc'}
    df4 = df4.rename(dictionnaire, axis=1)
    
    st.subheader("4) Données de surface couverte par les terres agricoles en pourcentage")
    st.write("##")
    year_options_4 = df4["Year"].unique().tolist()
    year_4 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_4, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018], key="4")
    country_options_4 = df4["Country_Name"].unique().tolist()
    country_4 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_4,["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovenia","Spain","Sweden","Switzerland","United Kingdom"])

    df4 = df4[df4["Country_Name"].isin(country_4)]
    df4 = df4[df4["Year"].isin(year_4)]
    st.write("##")
    st.dataframe(df4.head(10))
    st.write("Évolution de la surface couverte par les terres agricoles en pourcentage (source Global Forest Watch – données de 1990 à 2020)")
    fig4 = px.box(df4,x="Year",y="terres_agricoles_perc")
    fig4_1 = px.line(df4,x="Year",y="terres_agricoles_perc", color="Country_Code")
   
    st.plotly_chart(fig4)

    st.write ("On constate une forte disparité de la couverture de la surface agricole des différents pays inclus dans le dataset. La majorité des pays se situent autour de 50%. On constate également une lente érosion des valeurs de la surface couverte par les terres agricoles au fil des années incluses dans le dataset.")

    st.plotly_chart(fig4_1)
    
    st.write("##")
    st.write("##")
    





    #Dataset 5 sur la variable target utilisée pour la modélisation

    df5 = pd.read_csv('clean_PourExploration_Streamlit.csv')
    df5 = df5.drop(["surface_pct","country_global_emissions","CO2_aviation_sum","precipitation_m3","variation_temp_deg","terres_agricoles_perc","Aviation_perc"], axis=1)


    
    
    st.subheader("5) Données de flux carbone")
    st.write("##")
    year_options_5 = df5["Year"].unique().tolist()
    year_5 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_5, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020], key="5")
    country_options_5 = df5["Country_Name"].unique().tolist()
    country_5 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_5,["Austria","Belgium","Bulgaria","Croatia","Denmark","Estonia","Finland","France"])

    df5 = df5[df5["Country_Name"].isin(country_5)]
    df5 = df5[df5["Year"].isin(year_5)]
    st.write("##")
    st.dataframe(df5.head(10))
    st.write("Évolution des flux de carbone (source Global Forest Watch – données de 2000 à 2020)")
    fig5 = px.box(df5,x="Country_Name",y="net_carbon_stock_change", color="Country_Name")
    fig5_1 = px.box(df5,x="Year",y="net_carbon_stock_change")
   
    st.plotly_chart(fig5)
    st.write ("On constate encore une fois une disparité important des capacités d’absorption (les valeurs négatives correspondent aux capacités d’absorption en équivalent CO2 de la surface forestière de chaque pays.")

    st.write ("Nous avons sélectionné cette variable comme la variable cible que nous allons utiliser dans la modélisation. Nous souhaitons vérifier comment évolue cette variable dans les modèles que nous allons utiliser.")
    st.plotly_chart(fig5_1)
    
    

    st.write("##")
    st.write("##")




    #Dataset 6 sur les émissions de CO2 rejettées par l'aviation

    df6 = pd.read_csv('clean_aviation.csv')
    
    st.subheader("6) Données d'émission de CO2 induite par l'aviation")
    st.write("##")
    year_options_6 = df6["Year"].unique().tolist()
    year_6 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_6, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019], key="6")
    country_options_6 = df6["Country_Name"].unique().tolist()
    country_6 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_6,["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","United Kingdom"])

    df6 = df6[df6["Country_Name"].isin(country_6)]
    df6 = df6[df6["Year"].isin(year_6)]
    st.write("##")
    st.dataframe(df6.head(10))
    st.write("Évolution des émissions induites par l’aviation en Europe (source EUROSTAT 2021 – données de 1990 à 2019)")
    fig6 = px.box(df6,x="Country_Name",y="CO2_million_sum", color="Country_Name")
    fig6_1 = px.line(df6,x="Year",y="CO2_million_sum", color="Country_Name")
   
    st.plotly_chart(fig6)

    st.write ("On constate que les principaux pays contribuant aux émissions générées par l’aviation sont la France, l’Allemagne, l’Italie, les Pays-Bas, l’Espagne et surtout le Royaume-Uni. Cela correspond principalement à l’existence dans ces pays des principaux hubs pour les vols à destination des Etats-Unis (Royaume-Uni, Pays-Bas, Allemagne) mais également d’Afrique, pays du Golfe et d’Océanie (France, Italie) ou encore Amérique Latine (Espagne).")

    st.write ("Ces vols longs courriers étant responsables de la grande majorité des émissions de CO2 (longs courriers représentent environ 6% des vols des compagnies à partir de l’Europe mais environ 50% des émissions ).")

    st.warning ("NB : Nous avons initialement décidé d’utiliser un dataset publié par EUROCONTROL mais nous avons changé pour un dataset EUROSTAT (plus de données historiques disponibles en libre accès même si la qualité des données disponibles est inégale entre les pays européens avant 2000).")

    st.plotly_chart(fig6_1)
    
    st.write ("On constate une augmentation des émissions liées à l’aviation (et ce en dépit d’une baisse des émissions moyennes par passager depuis 1990). La baisse des émissions moyennes par passager n’est en effet pas suffisante pour compenser la croissance du trafic. ")

    st.write("##")
    st.write("##")





    #Dataset 7 sur les émissions de CO2 globales

    df7 = pd.read_csv('clean_eur28_global.csv')
    
    st.subheader("7) Données d'émission globale en Unio-Européenne et au Royaume-Uni")
    st.write("##")
    year_options_7 = df7["Year"].unique().tolist()
    year_7 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_7, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019], key="7")
    country_options_7 = df7["Country_Name"].unique().tolist()
    country_7 = st.multiselect("Choisissez un ou plusieurs pays :", country_options_7,["Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden","United Kingdom"])

    df7 = df7[df7["Country_Name"].isin(country_7)]
    df7 = df7[df7["Year"].isin(year_7)]
    st.write("##")
    st.dataframe(df7.head(10))
    st.write("Évolution des émissions globales en E.U et Royaume Uni (source EUROSTAT 2021 – données de 1990 à 2019)")
    fig7 = px.box(df7,x="Country_Name",y="EU28_CO2_emissions", color="Country_Name")
    fig7_1 = px.line(df7,x="Year",y="EU28_CO2_emissions", color="Country_Name")
    fig7_2 = px.box(df7,x="Year",y="EU28_CO2_emissions")
   
    st.plotly_chart(fig7)

    st.write ("On constate une corrélation entre les pays émettant le plus de CO2 et ceux ayant le plus de CO2 émis par le transport aérien.")

    st.write ("Quelques commentaires cependant, on observe que l’Allemagne est très significativement le pays émettant le plus de CO2, la Pologne y est également en 6ème position. Les disparités que l’on observe sont principalement dues au mix énergétique propre à chaque pays. Ainsi, une part très importante de l’électricité produite en Pologne l’est par des centrales à charbon , ce qui est également valable (bien que dans une moindre mesure) pour l’Allemagne.")

    st.plotly_chart(fig7_1)
    st.plotly_chart(fig7_2)
        
    
    st.write ("Il est intéressant de constater une relative diminution des émissions globales de CO2. On peut également observer une certaine corrélation entre les périodes de contraction économiques et de diminution des émissions de CO2. Ainsi, on constate que suite aux attentats de 2001 et la crise internationale et économique qui s’en est suivie, l’année 2002 les émissions globales de CO2 baissent légèrement.")

    st.write ("De manière similaire, à la suite de la crise financière ayant débuté à l’automne 2008, on constate une baisse notable des émissions de CO2 en 2009 (également liée à la crise de la dette souveraine en Europe).")

    st.write("##")
    st.write("##")





    #Dataset regroupant l'ensemble des données précédentes

    df = pd.read_csv('clean_PourExploration_Streamlit.csv')
    
    st.subheader("8) Données utilisées pour la Modélisation")
    st.write("##")
    year_options_8 = df["Year"].unique().tolist()
    year_8 = st.multiselect("Sélectionnez une ou plusieurs années :", year_options_8, [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020], key="8")
    country_options_8 = df["Country_Name"].unique().tolist()
    country_8 = st.multiselect("Choisissez un ou plisieurs pays :", country_options_8,['Austria','Belgium','Bulgaria','Croatia','Denmark','Estonia','Finland','France','Germany','Hungary','Ireland','Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands','Poland','Portugal','Romania','Slovenia','Spain','Sweden'])
    
    df = df[df["Country_Name"].isin(country_8)]
    df = df[df["Year"].isin(year_8)]
    st.write("##")
    st.dataframe(df.head(10))
    st.write("Il s'agit du dataset regroupant la totalité des données présentées précédemment, et qui sera utilisé lors de l'étape Modélisation.")

    fig8_2 = go.Figure()
    fig8_2.add_trace(go.Bar(x=df["Year"], y=df["CO2_aviation_sum"], name="CO2_aviation_sum"))
    fig8_2.add_trace(go.Bar(x=df["Year"], y=df["country_global_emissions"], name="country_global_emissions"))

    fig8_1 = px.line(df,x="Year",y="country_global_emissions", color="Country_Name")

    fig8_3 = px.line(df,x="Year",y="CO2_aviation_sum", color="Country_Name")

    

    
    #px.line_chart(df,x="Year",y="CO2_aviation_sum", color="Country_Name")

    
    #st.plotly_chart(fig8_2)
    st.plotly_chart(fig8_1)
    st.write ("On constate en croisant les différentes données qu’on observe une émissions significative des émissions moyennes liées à l’aviation par pays sur la période observée.  Dans le même temps, on peut constater une baisse des émissions moyennes globales de CO2.")
    st.plotly_chart(fig8_3)
    
    
    

    

    

    