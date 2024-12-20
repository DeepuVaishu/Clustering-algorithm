
# Clustering of Iris Dataset

This project implements clustering techniques (KMeans and Hierarchical Clustering) on the Iris dataset using Python. The goal is to apply both clustering algorithms to the Iris dataset, compare the results, and visualize the clusters. The notebook also demonstrates how to use the **Elbow Method** to find the optimal number of clusters for KMeans.

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Steps and Methodology](#steps-and-methodology)
   - [1. Loading and Preprocessing the Data](#loading-and-preprocessing-the-data)
   - [2. KMeans Clustering](#kmeans-clustering)
   - [3. Hierarchical Clustering](#hierarchical-clustering)
5. [Results and Observations](#results-and-observations)
6. [Conclusion](#conclusion)
7. [License](#license)

---

## **Project Overview**

The Iris dataset consists of 150 samples from three species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The dataset is well-suited for clustering tasks because it has well-separated classes that can be identified by clustering algorithms.

In this project, we:
1. Preprocess the Iris dataset by checking for missing values, duplicates, and scaling the features.
2. Implement KMeans and Hierarchical (Agglomerative) clustering to group the data into three clusters.
3. Use the **Elbow Method** to determine the optimal number of clusters for KMeans.
4. Visualize the clusters for both KMeans and Hierarchical clustering.
5. Plot a dendrogram to visualize the hierarchical clustering process.

---

## **Installation**

To run the notebook, you'll need to have the following Python packages installed:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- scipy

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## **Usage**

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/Clustering-Iris-Dataset.git
```

2. Navigate to the project directory:

```bash
cd Clustering-Iris-Dataset
```

3. Open the Jupyter notebook `Clustering Algorithm.ipynb` in Jupyter Notebook:

```bash
jupyter notebook Clustering Algorithm.ipynb
```

4. Follow the steps in the notebook to load, preprocess the data, and run the clustering algorithms.

---

## **Steps and Methodology**

### 1. **Loading and Preprocessing the Data**

In this step, the following operations are performed:
- Load the Iris dataset from the `sklearn` library.
- Remove the target column (`species`) since it’s not used in clustering.
- Check for missing values, duplicates, and visualize outliers using boxplots.
- Scale the features using `StandardScaler` to standardize the dataset.

### 2. **KMeans Clustering**

- **KMeans Clustering** is applied to group the data into three clusters (since there are three species in the Iris dataset).
- The **Elbow Method** is used to find the optimal number of clusters by plotting inertia (sum of squared distances to centroids).
- The clusters are visualized using a scatter plot, with each cluster assigned a unique color.

### 3. **Hierarchical Clustering**

- **Agglomerative (Hierarchical) Clustering** is used, which is a bottom-up approach to clustering.
- A **dendrogram** is plotted to visualize the merging of clusters and determine the optimal cutoff for clustering.
- The dendrogram also includes a horizontal threshold line that indicates the distance at which clusters are formed.

---

## **Results and Observations**

### KMeans Clustering:
- The KMeans algorithm successfully created 3 clusters.
- The **Elbow Method** helped identify the optimal number of clusters as 3, which corresponds to the true number of species in the Iris dataset.

### Hierarchical Clustering:
- The Hierarchical Clustering algorithm also produced 3 clusters.
- The dendrogram plot helped visualize the process of merging the clusters at different distance levels, confirming that 3 clusters is an appropriate choice.

Both KMeans and Hierarchical Clustering effectively grouped the Iris dataset into three distinct clusters.

---

## **Conclusion**

This project demonstrates the application of two clustering techniques on the Iris dataset: KMeans and Hierarchical Clustering. Both algorithms produced meaningful clusters that align with the actual species in the dataset. The **Elbow Method** was used to determine the optimal number of clusters for KMeans, and the **dendrogram** provided valuable insights into the merging process in Hierarchical Clustering.

The project provides a clear understanding of how to apply and interpret clustering algorithms on real-world data.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
