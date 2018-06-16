# -*- coding: utf-8 -*-
"""
@author: cordula eggerth

aufgabe 1 und 2 / uebung 4 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy.contrasts import Treatment


# -----------------------------------------------------------------------------
# DATEN AUS CSV-FILES EINLESEN & DATAFRAMES ANLEGEN
# -----------------------------------------------------------------------------
# adressen (spalten: PLZ, stadt)
adressen = pd.read_csv("C:/Users/cordu/Desktop/UE4/Adresse.csv", sep=";", decimal=",") 
adressen_df = pd.DataFrame(adressen)

# benutzer (spalten: UID, PLZ)
benutzer = pd.read_csv("C:/Users/cordu/Desktop/UE4/Benutzer.csv", sep=";", decimal=",") 
benutzer_df = pd.DataFrame(benutzer)

# features (spalten: UID, datum, variable, value)
features = pd.read_csv("C:/Users/cordu/Desktop/UE4/Features.csv", sep=";", decimal=",") 
features_df = pd.DataFrame(features)


# -----------------------------------------------------------------------------
# DATEN ZUSAMMENFUEHREN
# -----------------------------------------------------------------------------

# Join von benutzer und adressen auf "PLZ" 
joined_benutzer_adressen = pd.merge(benutzer, adressen, how="inner", 
                                    on= "PLZ")

# Join von joined_benutzer_adressen mit features auf "UID" 
joined_all = pd.merge(joined_benutzer_adressen, features, how="inner", 
                                    on= "UID")

# CHECK DATA TYPES OF DATAFRAME (DF)
joined_all.dtypes

 # joined_all ist das zusammengelegte data.frame, das alle informationen enthaelt
type(joined_all)
print(joined_all)


###############################################################################
# AUFGABE 1
###############################################################################

# 1.a. Werten Sie alle in den Daten vorkommenden Features (Feature 1 bis Feature 6) deskriptiv aus.
#      Bitte beachten Sie, dass sich mit den Städtenamen (Tabelle Adresse) und den Datumsangaben 6 verschiedene 
#      Gruppen bilden. 
#      Berechnen Sie den Mittelwert, den Median, die Standardabweichung, den minimalen und maximalen Wert sowie 
#      die 25% und 75% Quantile. 
#      Erstellen Sie für jedes der Features je eine Grafik mit 6 Histogrammen für die Gruppen (horizontal die 3 
#      Städte, vertikal das Datum). 
#      Hinweis: verwenden Sie die subplots Funktion aus der matplotlib Library, 
#      Beispiel: https://matplotlib.org/examples/pylab_examples/subplots_demo.html
# 1.b. Gibt es fehlende Werte in dem Datensatz? Wenn ja, wie viele? Können Sie diese Daten imputieren? 
#      Falls Sie die Daten nicht imputieren können, dann entfernen Sie unvollständige Datensätze aus dem 
#      Analysebestand.
# 1.c. Visualisieren Sie die Korrelationsmatrizen.
#      (z.B. unter Verwendung der matshow Funktion aus der matplotlib Bibliothek).


 # DATEN IN FEATURE 1 BIS 6 TRENNEN und anzahl der beobachtungen ueberpruefen

daten_feature_1 = joined_all[ joined_all["variable"] == "Feature 1"]
daten_feature_1.UID.count() # 299 beobachtungen (bzw. zeilen)

daten_feature_2 = joined_all[ joined_all["variable"] == "Feature 2"]
daten_feature_2.UID.count() # 300 beobachtungen (bzw. zeilen)

daten_feature_3 = joined_all[ joined_all["variable"] == "Feature 3"]
daten_feature_3.UID.count() # 299 beobachtungen (bzw. zeilen)

daten_feature_4 = joined_all[ joined_all["variable"] == "Feature 4"]
daten_feature_4.UID.count() # 300 beobachtungen (bzw. zeilen)

daten_feature_5 = joined_all[ joined_all["variable"] == "Feature 5"]
daten_feature_5.UID.count() # 300 beobachtungen (bzw. zeilen)

daten_feature_6 = joined_all[ joined_all["variable"] == "Feature 6"]
daten_feature_6.UID.count() # 300 beobachtungen (bzw. zeilen)

anzahl_UID = daten_feature_6.UID.count()

# ANMERKUNG:
# fuer jeden benutzer (UID) wurden scheinbar 2 beobachtungen pro feature gemacht
# es fehlt aber bei feature 3 eine beobachtung und bei feature 1 eine beobachtung, 
# damit es tatsaechlich 2 beobachtungen pro feature waeren pro benutzer (UID)
# die werte fuer den betroffenen benutzer (UID) kann man bis auf den value herausfinden.
# fuer den value koennte man eine schaetzung machen. es gibt hier aber keine vorgabe, 
# wie man die schaetzung durchfuehren sollte.


# UNVOLLSTAENDIGE BEOBACHTUNGEN:
# finde UID, deren beobachtungen unvollstaendig sind
    # bezueglich feature 1
unvollstaendige_UID = "a"
for i in range(0,anzahl_UID):
    if( (daten_feature_1["UID"] == benutzer["UID"][i]).sum() < 2):
        unvollstaendige_UID = benutzer["UID"][i]
        break
# resultat: UID 30 ist unvollstaendig bzgl. feature 1, 
#           weil es gibt nur 1 beobachtung davon
(daten_feature_1["UID"]==30).sum()
   
    # bezueglich feature 3
unvollstaendige_UID_f3 = "a"
for i in range(0,anzahl_UID):
    if( (daten_feature_3["UID"] == benutzer["UID"][i]).sum() < 2):
        unvollstaendige_UID_f3 = benutzer["UID"][i]
        break
# resultat: UID 130 ist unvollstaendig bzgl. feature 3, 
#           weil es gibt nur 1 beobachtung davon
(daten_feature_3["UID"]==130).sum()     


# LISTE VON DATEN_FEATURE DFs
liste_featuresdaten = [daten_feature_1, daten_feature_2, daten_feature_3,
                       daten_feature_4, daten_feature_5, daten_feature_6]
    

# UMGANG MIT UNVOLLSTAENDIGEN BEOBACHTUNGEN: 
# zur sicherheit, dass keine datenverfaelschungen entstehen, 
# werden die beobachtungen UID 30 und 130 weggelassen

# FUNKTION: entferne unvollstaendige beobachtungen
def entferneUnvollstaendigeBeobachtungen(liste_featuresdaten):
    """
    funktion entfernt unvollstaendige beobachtungen
    param: liste_featuresdaten
    returnwert: daten_feature_cleaned 
    """ 
    
    for i in range(0, len(liste_featuresdaten)):
        daten_feature_i = liste_featuresdaten[i]
        daten_feature_i_cleaned = daten_feature_i[ daten_feature_i["UID"] != unvollstaendige_UID ]
        daten_feature_i_cleaned = daten_feature_i_cleaned[ daten_feature_i_cleaned["UID"] != unvollstaendige_UID_f3 ]
        liste_featuresdaten[i] = daten_feature_i_cleaned


    return liste_featuresdaten


# aufruf der funktion entferneUnvollstaendigeBeobachtungen für die gesamtliste
    liste_featuresdaten = entferneUnvollstaendigeBeobachtungen(liste_featuresdaten)


# CHECK: ob alle listenelemente nun gleiche anzahl an beobachtungen haben 
# (hier sollten es 296 beob. pro daten_feature nach entfernung der 2 unvollstaendigen UIDs sein)

for i in range(0,len(liste_featuresdaten)):
    print(liste_featuresdaten[i]["UID"].count())
    # ANMERKUNG: ja, alle haben nun 296 beobachtungen
        

# unvollstaendige beobachtungen aus joined_all entfernen (gesamtdaten ohne featurestrennung)
joined_all = joined_all[ joined_all["UID"] != unvollstaendige_UID ]
joined_all = joined_all[ joined_all["UID"] != unvollstaendige_UID_f3 ]

# CHECK, ob 296*6 elemente:
len(joined_all) == 296*6  # ja, ist korrekt



# AGGREGIERTE STATISTIKEN BERECHNEN
# aggregation definieren:
# Quelle: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
aggregierte_deskriptiv = {
    'value': { # spalte "value"
        'Mean': 'mean',    
        'Median': 'median',  
        'Standard Deviation': 'std', # sample standard deviation
        'Min': 'min',  
        'Max': 'max', 
        'Quantile25': lambda q: q.quantile([0.25]), 
        'Quantile75': lambda q: q.quantile([0.75]), 
    }
}


# DATEN GRUPPIEREN: (nach variable, Stadt und Datum in 6 gruppen pro feature)
    # GRUPPEN:
    # G1: Graz - 20160813T00:00Z
    # G2: Graz - 20180416T00:00Z  
    # G3: Salzburg - 20160813T00:00Z
    # G4: Salzburg - 20180416T00:00Z
    # G5: Wien - 20160813T00:00Z
    # G6: Wien - 20180416T00:00Z
deskriptiveStatistik_aggregiert = joined_all.groupby(['variable', 'Stadt', 'Datum']).agg(aggregierte_deskriptiv)    
print(deskriptiveStatistik_aggregiert)


## ---------------------------------------------------------------------------------
## VISUALISIERUNG (HISTOGRAMME FUER DIE 6 GRUPPEN)    
## ---------------------------------------------------------------------------------
   
# basisdaten (liste mit jeweiligen featuresdaten)
liste_featuresdaten

# FUNKTION: plot fuer alle gruppen eines features machen
def plotGroupsPerFeature(daten_feature, featurenumber):
    """
    funktion erstellt fuer das uebergebene daten_feature die gruppenplots
    param: daten_feature
    param: featurenumber
    """ 
    # gruppen bilden
    gruppe1 = daten_feature[ (daten_feature["Stadt"]=="Graz") & (daten_feature["Datum"]=="20180416T00:00Z") ]
    gruppe2 = daten_feature[ (daten_feature["Stadt"]=="Graz") & (daten_feature["Datum"]=="20160813T00:00Z") ]
    gruppe3 = daten_feature[ (daten_feature["Stadt"]=="Salzburg") & (daten_feature["Datum"]=="20180416T00:00Z") ]
    gruppe4 = daten_feature[ (daten_feature["Stadt"]=="Salzburg") & (daten_feature["Datum"]=="20160813T00:00Z") ]
    gruppe5 = daten_feature[ (daten_feature["Stadt"]=="Wien") & (daten_feature["Datum"]=="20180416T00:00Z") ]
    gruppe6 = daten_feature[ (daten_feature["Stadt"]=="Wien") & (daten_feature["Datum"]=="20160813T00:00Z") ]
    
    liste_fuerPlot = [gruppe2, gruppe4, gruppe6, gruppe1, gruppe3, gruppe5] 
    gruppennamen_fuerPlot = ["Graz - 2016", "Salzburg - 2016", "Wien - 2016", 
                             "Graz - 2018", "Salzburg - 2018", "Wien - 2018"]
    # reihenfolge der gruppen in plot: 2 4 6 1 3 5
    
        
    # subplots anfertigen
    # Quelle: https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html  
    fig=plt.figure(figsize=(15, 15))
    fig.suptitle("Feature " + str(featurenumber+1), fontsize=20) 
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        plt.xticks(rotation=90)
        max_xticks = 20
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)
        plt.hist(liste_fuerPlot[i-1]["value"], color = "deepskyblue", alpha=0.4, histtype='bar', ec='black')
        plt.subplot(2, 3, i).set_title(gruppennamen_fuerPlot[i-1])
        plt.subplots_adjust(hspace=0.5)
        
# CHECK: fuer daten_feature_2, ob funktion korrektes ergebnis liefert
plotGroupsPerFeature(daten_feature_4, 3)


# FUNKTION plotGroupsPerFeature fuer alle features aufrufen
for i in range(0,len(liste_featuresdaten)):
    plotGroupsPerFeature(liste_featuresdaten[i], i)
    
    

## ---------------------------------------------------------------------------------
## VISUALISIERUNG (KORRELATIONSMATRIZEN)    
## ---------------------------------------------------------------------------------
       
### GESAMT-KORRELATIONSMATRIX BASIEREND AUF DEN VALUES DER FEATURES (i.e werte der variablen) 
    
# basisdaten
# dataframe fuer values der features 1 bis 6 anlegen (basierend auf daten nach
# entfernen der unvollstaendigen beobachtungen)
liste_featuresdaten

feature_values  = {'f1': list(liste_featuresdaten[0]["value"]),
        'f2': list(liste_featuresdaten[1]["value"]),
        'f3': list(liste_featuresdaten[2]["value"]),
        'f4': list(liste_featuresdaten[3]["value"]),
        'f5': list(liste_featuresdaten[4]["value"]),
        'f6': list(liste_featuresdaten[5]["value"]),
        }

labels = list(liste_featuresdaten[0]["UID"])

df_values = pd.DataFrame(feature_values, index=labels)

# KORRELATION MATRIX BERECHNEN
correlation_matrix = df_values.corr()
print(correlation_matrix)

# KORRELATION MATRIX VISUALISIEREN
# Quelle: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,6,1)
names = ["f1", "f2", "f3", "f4", "f5", "f6"]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# ÜBERBLICK ÜBER df_values MITTELS SCATTERMATRIX
pd.scatter_matrix(df_values, figsize=(6, 6))
plt.show()



## ------------------------------------------------------
## KORRELATION PRO GRUPPE (alle features betrachtet)
## ------------------------------------------------------
# basisdaten
joined_all

# gruppen bilden
g1 = joined_all[ (joined_all["Stadt"]=="Graz") & (joined_all["Datum"]=="20180416T00:00Z") ]
g2 = joined_all[ (joined_all["Stadt"]=="Graz") & (joined_all["Datum"]=="20160813T00:00Z") ]
g3 = joined_all[ (joined_all["Stadt"]=="Salzburg") & (joined_all["Datum"]=="20180416T00:00Z") ]
g4 = joined_all[ (joined_all["Stadt"]=="Salzburg") & (joined_all["Datum"]=="20160813T00:00Z") ]
g5 = joined_all[ (joined_all["Stadt"]=="Wien") & (joined_all["Datum"]=="20180416T00:00Z") ]
g6 = joined_all[ (joined_all["Stadt"]=="Wien") & (joined_all["Datum"]=="20160813T00:00Z") ]

# gruppenliste
gruppenliste = [g1,g2,g3,g4,g5,g6]
  
# FUNKTION: plot fuer jeweilige gruppe korrelation zwischen den features 1-6
def plotCorrelationPerGroup(groupData, groupNr):
    """
    funktion visualisiert fuer das uebergebene groupdata die korrelation
    param: groupData
    param: groupNr
    """ 
    
    f1 = groupData[ groupData["variable"]=="Feature 1" ]["value"]
    f2 = groupData[ groupData["variable"]=="Feature 2" ]["value"]
    f3 = groupData[ groupData["variable"]=="Feature 3" ]["value"]
    f4 = groupData[ groupData["variable"]=="Feature 4" ]["value"]
    f5 = groupData[ groupData["variable"]=="Feature 5" ]["value"]
    f6 = groupData[ groupData["variable"]=="Feature 6" ]["value"]
    
    data = {'f1': list(f1), 
            'f2': list(f2),
            'f3': list(f3),
            'f4': list(f4),
            'f5': list(f5),
            'f6': list(f6)
            }
        
    df_values = pd.DataFrame(data, columns=['f1','f2','f3','f4','f4','f5'])
    
    # KORRELATION MATRIX BERECHNEN
    correlation_matrix = df_values.corr()
    print(correlation_matrix)
    
    # KORRELATION MATRIX VISUALISIEREN
    # Quelle: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.subplot().set_title("Korrelation Gruppe " + str(groupNr+1))
    cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,6,1)
    names = ["f1", "f2", "f3", "f4", "f5", "f6"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    


# FUNKTION plotCorrelationPerGroup fuer alle gruppen aufrufen
for i in range(0,len(gruppenliste)):
    plotCorrelationPerGroup(gruppenliste[i], i)
    


# CHECKS:
groupData=gruppenliste[1]
c=groupData[ groupData["variable"]=="Feature 1" ]["value"]
type(c)
list(c)
f1 = groupData[ groupData["variable"]=="Feature 1" ]["value"]
f2 = groupData[ groupData["variable"]=="Feature 2" ]["value"]
f3 = groupData[ groupData["variable"]=="Feature 3" ]["value"]
f4 = groupData[ groupData["variable"]=="Feature 4" ]["value"]
f5 = groupData[ groupData["variable"]=="Feature 5" ]["value"]
f6 = groupData[ groupData["variable"]=="Feature 6" ]["value"]

data = {'f1': list(f1), 
        'f2': list(f2),
        'f3': list(f3),
        'f4': list(f4),
        'f5': list(f5),
        'f6': list(f6)
        }
    
df_values = pd.DataFrame(data, columns=['f1','f2','f3','f4','f4','f5'])
df_values



###############################################################################
# AUFGABE 2
###############################################################################

# 2.a. Erzeugen Sie eine abgeleitete Variable aus der Summe von Feature 5 und Feature 6

# neue abgeleitete variable aus summe von f5 und f6 bilden
variable_sum56 = list()
for i in range(0, len(list(liste_featuresdaten[5]["value"]))):
    variable_sum56.append( list(liste_featuresdaten[4]["value"])[i] + list(liste_featuresdaten[5]["value"])[i] )
len(variable_sum56)

# neues dataframe feature_values_2 erstellen fuer die analyse
feature_values_2  = {'f1': list(liste_featuresdaten[0]["value"]),
        'f2': list(liste_featuresdaten[1]["value"]),
        'f3': list(liste_featuresdaten[2]["value"]),
        'f4': list(liste_featuresdaten[3]["value"]),
        'fNEU': variable_sum56,
        'stadt': list(liste_featuresdaten[5]["Stadt"]),
        'datum': list(liste_featuresdaten[5]["Datum"])        
        }

labels = list(liste_featuresdaten[1]["UID"])

df_variables_2 = pd.DataFrame(feature_values_2, index=labels)


# 2.b. Gibt es Korrelationen zwischen den verbleibenden Variablen und der neuen abgeleiteten Variable?

df_fuerKorr = df_variables_2.loc[:, ['f1', 'f2', 'f3', 'f4', 'fNEU',]]    
     
# KORRELATION MATRIX BERECHNEN
correlation_matrix_neu = df_variables_2.corr()
print(correlation_matrix_neu)
    
# KORRELATION MATRIX VISUALISIEREN
# Quelle: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlation_matrix_neu, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
names = ["f1", "f2", "f3", "f4", "fNEU"]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# ÜBERBLICK ÜBER df_values MITTELS SCATTERMATRIX
pd.scatter_matrix(df_variables_2, figsize=(6, 6))
plt.show()



# 2.c. Modellieren Sie die abgeleitete Variable mit einem linearen Modell.
# 2.d. Welche Variablen sind im Modell sinnvoll, wie gehen Sie mit den kategoriellen Variablen um?
# 2.e. Beschreiben Sie Ihre Modellierungsergebnisse und erzeugen Sie Grafiken um Ihre Ergebnisse zu dokumentieren.
# 2.f. Welche Modellierungsmethode verwenden Sie und warum haben Sie sich für dieses Modell entschieden?
 
# LINEARES MODELL MIT MODELLIERUNG ALS ADDITIVEM ZUSAMMENHANG
# zielvariable ist fNEU
# predictorvariablen sind f1, f2, f3 und f4
results = smf.ols('fNEU ~ f1+f2+f3+f4', data=df_variables_2).fit()
print(results.summary())
   # ANMERKUNG: 
   # f1, f2, f3 signifikant
   # f4 nicht signifikant
   # (adj.) R squared ca. 10%


# LINEARES MODELL MIT BERUECKSICHTIGUNG DER INTERAKTIONEN
# zielvariable ist fNEU
# predictorvariablen sind f1, f2, f3 und f4
results2 = smf.ols('fNEU ~ f1*f2*f3*f4', data=df_variables_2).fit()
print(results2.summary())
   # ANMERKUNG: 
   # interaktionen nicht signifikant
   # (adj.) R squared ca. 10% bzw. R-squared 15%


# REGRESSION-DIAGNOSE

# QQ plot
resid = results.resid
fig  = sm.qqplot(resid, line="s")
plt.show()
# interpretation:
# im quantile-quantile-plot (i.e. theoretische
# quantile vs. empirische quantile) sollten die punkte
# im falle der erfuellten normalverteilungsannahme
# bzgl. der fehlerterme auf der gerade liegen.
# im vorliegenden fall is dies im intervall von ca. [-2,2]
# so, aber an den enden nicht

# influence plot 
sm.graphics.influence_plot(results, criterion="Cooks")
plt.show()
# interpretation:
# punkte, die weit rechts oben (od. rechts unten) liegen, sind 
# einflussreiche punkte auf die regression
# hier: die punkte liegen alle bzgl. leverage mittig, aber teilweise
# bzgl. studentized residuals (outlyingness) über die ganze bandbreite 
# reichend
# influence wird durch punktgroesse dargestellt 
# hier: eine reihe von punkten wird vergleichsweise groß dargestellt 
# (e.g. 108, 31, 122 etc.)


# VERSUCH DER MODELLIERUNG MIT ROBUSTER METHODE 
# (unter verwendung des huber estimator)
# robuste methoden, versuchen mit outliern umzugehen auf eine art und weise, 
# dass diese nicht so stark ins gewicht fallen in den regressionsresultaten
results3 = smf.RLM.from_formula(formula='fNEU ~ f1+f2+f3+f4', data=df_variables_2,
                                M=sm.robust.norms.HuberT()).fit()
print(results3.summary())
# interpretation:
# f1, f2, f3 sind signifikant
# ergebnis hat sich im vgl. zu den 2 vorigen modellen
# nicht sehr stark verändert


# UMGANG MIT KATEGORIELLEN VARIABLEN:
# LINEARES MODELL MIT MODELLIERUNG ALS ADDITIVEM ZUSAMMENHANG UND INTERAKTIONEN
# zielvariable ist fNEU
# predictorvariablen sind f1, f2, f3, stadt, datum
# dummy-codierung (bzw. treatment-codierung) der kategoriellen variablen)
results4 = smf.ols('fNEU ~ (f1+f2+f3+f4)*C(stadt)*C(datum)', data=df_variables_2).fit()
print(results4.summary())
   # ANMERKUNG: 
   # f2:C(datum)[T.20180416T00:00Z] signifikant
   # f3 signifikant 
   # f4:C(stadt)[T.Salzburg] signifikant
   # erklärte varianz wurde auf R-square=36% und adj. R-squared=29% erhöht
   # dieses modell behandelt bisher am besten die zusammenhaenge in den daten
   
   

