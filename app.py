import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.title("Bank Customer Segmentation using KMeans")

file = st.file_uploader("Upload BankChurners Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    df.drop([
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
], axis=1, inplace=True)

    q1 = df['Total_Trans_Amt'].quantile(0.25)
    q3 = df['Total_Trans_Amt'].quantile(0.75)

    IQR = q3 - q1

    lower = q1 - (0.75 * IQR)
    upper = q3 + (0.75 * IQR)

    df = df[(df['Total_Trans_Amt'] >= lower) &
            (df['Total_Trans_Amt'] <= upper)]

    columns = ['CLIENTNUM','Attrition_Flag','Gender',
               'Education_Level','Marital_Status',
               'Income_Category','Card_Category']

    en = pd.get_dummies(df[columns], dtype=int)

    df_en = pd.concat([df.drop(columns, axis=1), en], axis=1)

    df_en.drop('CLIENTNUM', axis=1, inplace=True)

    scaler = StandardScaler()

    df_norm = scaler.fit_transform(df_en)

    pca = PCA(n_components=2)

    pca_comp = pca.fit_transform(df_norm)

    pca_df = pd.DataFrame(pca_comp, columns=["PC1","PC2"])

    kmeans = KMeans(n_clusters=3)

    labels = kmeans.fit_predict(pca_df)

    pca_df["cluster"] = labels

    st.write("Clustered Data")
    st.dataframe(pca_df.head())

    fig = plt.figure(figsize=(10,6))

    sns.scatterplot(data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue="cluster")

    st.pyplot(fig)