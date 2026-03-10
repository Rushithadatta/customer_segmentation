# Customer Segmentation using Machine Learning

## Overview

Customer segmentation is a data analysis technique used to divide customers into distinct groups based on their purchasing behavior, demographics, or engagement patterns. Businesses use segmentation to better understand customer needs and create personalized marketing strategies.

This project implements **unsupervised machine learning techniques** to identify meaningful customer groups from a retail dataset. The goal is to help businesses understand different types of customers and design targeted marketing strategies for each segment.

The model groups customers based on spending patterns and income characteristics using clustering algorithms.

---

## Objectives

The main objectives of this project are:

- Perform **data preprocessing and feature preparation** for clustering
- Apply **K-Means clustering** to segment customers into distinct groups
- Visualize customer segments using data visualization techniques
- Build an interactive interface for exploring customer clusters

---

## Dataset

The dataset used in this project represents **customer shopping behavior in a retail environment**.

### Features

| Feature | Description |
|------|------|
| CustomerID | Unique ID for each customer |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousand dollars |
| Spending Score (1-100) | Score assigned based on spending behavior |

This dataset helps identify purchasing patterns and income-based segmentation.

---

## Machine Learning Technique

### K-Means Clustering

K-Means is an **unsupervised learning algorithm** used to group similar data points into clusters.

The algorithm works in the following steps:

1. Select number of clusters **K**
2. Randomly initialize cluster centroids
3. Assign each data point to the nearest centroid
4. Recalculate centroids based on cluster members
5. Repeat until convergence

The **Elbow Method** is used to determine the optimal number of clusters.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer_segmentation.git
cd customer_segmentation
```
## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```
Training the Model

Run the training script:

python train_model.py

### This script will:

Load the dataset

Preprocess customer features

Train the K-Means clustering model

Save the trained model

Running the Application

### Start the Streamlit application:
``` bash
streamlit run app.py
```

### The application allows users to:

- Visualize customer clusters

- Explore segmentation patterns

- Understand customer behavior distributions

- Results

### The clustering algorithm groups customers into meaningful segments such as:

- High income – high spending customers

- High income – low spending customers

- Low income – high spending customers

- Low income – low spending customers

These insights help businesses design targeted marketing campaigns and improve customer engagement.

### Applications

Customer segmentation is widely used in:

- Personalized marketing

- Customer relationship management

- Retail analytics

- Recommendation systems

- Customer behavior analysis

### Future Improvements

Potential improvements include:

- Using DBSCAN or Hierarchical Clustering

- Adding Dimensionality Reduction (PCA)

- Deploying the application on Streamlit Cloud

- Integrating real-time customer data pipelines

### Author

Rushitha Datta
Computer Science Student | Machine Learning Enthusiast

Interested in Machine Learning, Data Science, and Algorithmic Systems
