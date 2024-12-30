# Wine Classification using Machine Learning

## **Overview**
This project applies machine learning techniques to classify **Red** and **White** wines based on the **UCI Wine Quality Dataset**. The dataset combines red and white wine data, and the classification task is to distinguish between the two wine types. The performance of three machine learning algorithms is evaluated: 
- **Support Vector Machine (SVM)** 
- **K-Nearest Neighbors (KNN)** 
- **Naive Bayes (NB)**

The dataset consists of **5320 samples**, each with **13 features**, and the wine type (`Red` or `White`) as the label.

---

## **Dataset Preparation**
1. **Data Sources**:
   - Two separate datasets were combined:
     - **Red Wine**: Labeled as `0`
     - **White Wine**: Labeled as `1`

2. **Preprocessing**:
   - Added a new column, **"Wine Type"**, to indicate the label.
   - Combined the datasets along the rows (`axis=0`).
   - Checked for duplicates and missing values.
   - Normalized the data to improve training efficiency.
   - Split the dataset into:
     - **80% for Training**: 4256 samples.
     - **20% for Testing**: 1064 samples.

---

## **Models and Results**
### **1. Support Vector Machine (SVM)**
- **Description**:
  - A hyperplane-based algorithm that achieved the highest performance among the three models.
- **Hyperparameter Tuning**:
  - Optimized using **Grid Search** and validated using **5-Fold Cross-Validation**.
- **Performance**:
  - **Accuracy**: 99.71%
  - **Precision**: 99.87%
  - **Recall**: 99.74%
  - **Training Time**: 0.479 seconds
  - **Testing Time**: 0.0036 seconds
  - **Confusion Matrix**:
    ```
    [288   1]
    [  2 773]
    ```

### **2. K-Nearest Neighbors (KNN)**
- **Description**:
  - A distance-based algorithm that performed competitively with SVM while being computationally faster during training.
- **Parameters**:
  - **Neighbors (k)**: 5
  - **Leaf Size**: 10
- **Performance**:
  - **Accuracy**: 99.24%
  - **Precision**: 99.86%
  - **Recall**: 99.09%
  - **Training Time**: 0.010 seconds
  - **Testing Time**: 0.0725 seconds
  - **Confusion Matrix**:
    ```
    [288   1]
    [  7 768]
    ```

### **3. Naive Bayes (NB)**
- **Description**:
  - A probabilistic algorithm that was the fastest among the three models but had slightly lower accuracy and recall.
- **Performance**:
  - **Accuracy**: 98.02%
  - **Precision**: 99.21%
  - **Recall**: 98.06%
  - **Training Time**: 0.0019 seconds
  - **Testing Time**: 0.0010 seconds
  - **Confusion Matrix**:
    ```
    [283   6]
    [ 15 760]
    ```

---

## **Evaluation Metrics**
1. **Accuracy**:
   - Measures the overall correctness of the model.
2. **Precision**:
   - Indicates the proportion of true positive predictions among all positive predictions.
3. **Recall**:
   - Measures the proportion of true positives correctly identified by the model.
4. **Confusion Matrix**:
   - Visualizes the counts of True Positives, True Negatives, False Positives, and False Negatives.
5. **ROC Curve** (for SVM):
   - The SVM ROC curve demonstrates its superior ability to distinguish between Red and White wines, with an optimal threshold selection.

---

## **Conclusion**
1. **Best Model**: 
   - **SVM** achieved the highest accuracy, precision, and recall, with the best confusion matrix and ROC curve.
2. **KNN**:
   - Performed comparably to SVM but had lower recall and longer testing time.
3. **Naive Bayes**:
   - Although the fastest, NB exhibited slightly lower performance in all metrics.
4. **Hyperparameter Tuning**:
   - Cross-validation and grid search significantly improved model performance by optimizing the parameters.
5. **SVM Insights**:
   - The SVM model demonstrated excellent class separation, making it the most suitable choice for this task despite higher computational costs.

---

## **How to Run the Project**
1. Clone the repository and navigate to the project directory.
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
