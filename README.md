# AI-ML-Algorithms

This repository contains implementations of various **Machine Learning algorithms** in Python. The goal is to build a comprehensive collection of ML techniques with clear, well-documented code examples and explanations to help learn and apply these algorithms effectively.

---

## Implementations

### TensorFlow

**Explanation:**
TensorFlow is a comprehensive, open-source platform developed by Google Brain for building and deploying machine learning and deep learning models. It uses a flexible architecture based on dataflow graphs, where computations are represented as nodes (operations) and edges (multi-dimensional data arrays called tensors) flow between them. TensorFlow supports both eager execution—where operations are evaluated immediately—and graph execution—where operations are compiled as a static computation graph for optimized performance.

TensorFlow’s core strength lies in its scalability across multiple platforms, from desktops and servers to mobile devices and edge hardware. It supports distributed computing, allowing models to be trained efficiently on large datasets using CPUs, GPUs, and specialized Tensor Processing Units (TPUs).

The platform integrates tightly with Keras, a high-level API that simplifies building neural networks, and provides tools for model training, evaluation, deployment, and optimization, including TensorFlow Lite for mobile and embedded devices, and TensorFlow Serving for production environments. AutoDifferentiation enables automatic computation of gradients, which is critical for training deep neural networks.

**Pros:**

- Highly scalable: runs on CPUs, GPUs, TPUs, and distributed environments.

- Rich ecosystem including Keras, TensorFlow Lite, TensorFlow.js, and TensorFlow Extended (TFX) for production.

- Supports flexible execution modes (eager and graph execution).

- Automatic differentiation simplifies gradient calculations during training.

- Extensive tooling for model building, debugging, and deployment.

**Cons:**

- Can have a steep learning curve due to its complexity and breadth.

- Debugging graph execution can be more challenging compared to eager mode or simpler frameworks.

- Large framework size and overhead may be excessive for very small tasks.

**Use Cases:**
TensorFlow is widely used for a variety of machine learning applications—from natural language processing and computer vision to time-series forecasting and reinforcement learning. It is suitable for both research and production, powering everything from academic experiments to large-scale industrial AI deployments including mobile apps, web services, and edge computing solutions.

---

### PyTorch

**Explanation:**
PyTorch is an open-source deep learning framework developed by Facebook’s AI Research (FAIR) lab. It is designed to provide flexibility, speed, and ease of use for both researchers and practitioners. At its core, PyTorch uses a dynamic computation graph (define-by-run), meaning the graph is built on the fly as operations are executed. This makes debugging, experimentation, and prototyping more intuitive compared to static graph frameworks.

PyTorch supports automatic differentiation through its `autograd` system, allowing seamless computation of gradients during backpropagation. It integrates well with Python and leverages popular libraries like NumPy, making it highly accessible to developers. The framework also supports deployment via **TorchScript** for optimized execution and **PyTorch Mobile** for running models on edge devices.

With strong GPU acceleration (CUDA integration) and a rapidly growing ecosystem, PyTorch has become a standard in academic research, and its production-ready tools like **TorchServe** and **ONNX support** have led to wide adoption in industry as well.

**Pros:**

- Dynamic computation graph makes debugging and experimentation intuitive.  
- Simple, Pythonic API that feels natural for Python developers.  
- Strong community and wide adoption in both research and industry.  
- Excellent GPU acceleration with CUDA.  
- Supports deployment via TorchScript, PyTorch Mobile, and ONNX.  

**Cons:**

- May be slightly slower than static graph frameworks in certain production use cases.  
- Smaller ecosystem compared to TensorFlow in terms of end-to-end tooling (though rapidly growing).  

**Use Cases:**
PyTorch is widely used in research and production for computer vision, natural language processing, reinforcement learning, generative models (GANs, VAEs), and scientific computing. Its flexibility and ease of use make it the preferred framework for researchers, while tools like TorchServe, TorchScript, and ONNX make it practical for deploying models in real-world applications.


---

### Linear Regression
**Explanation:**  
Linear Regression is a foundational supervised machine learning algorithm used for predicting a continuous target variable based on one or more input features. It assumes there is a linear relationship between the independent variables (features) and the dependent variable (target). The model fits a straight line (or hyperplane in multiple dimensions) that best represents the data by minimizing the sum of squared differences between actual and predicted values.

The equation for multiple linear regression is:  
\[
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
\]  
where \(\hat{y}\) is the predicted value, \(x_i\) are the features, and \(\theta_i\) are the coefficients learned from the data.

Linear regression is widely used because of its simplicity, interpretability, and efficiency. It provides intuitive insights into the relationship between predictors and the outcome, making it a great starting point for regression tasks.

**Pros:**  
- Easy to understand and explain.  
- Fast training and prediction, scalable to large datasets.  
- Provides explicit coefficients representing feature importance.  
- Performs well when the true relationship is approximately linear.

**Cons:**  
- Assumes linearity, which may not hold for complex data.  
- Sensitive to outliers that can disproportionately influence the fit.  
- Does not handle multicollinearity or interactions without additional processing.

**Use Cases:**  
Ideal for problems where the relationship between variables is expected to be linear, such as predicting house prices based on square footage, sales forecasting, or any scenario requiring numeric prediction from straightforward features.

---

### Logistic Regression
**Explanation:**  
Logistic Regression is a supervised learning algorithm used primarily for binary classification problems. Unlike linear regression, it outputs probabilities by applying the logistic (sigmoid) function to a linear combination of input features, squashing real values into the range [0,1]. This probability can then be thresholded (commonly at 0.5) to assign class labels.

The model estimates the odds of the positive class via:  
\[
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
\]  
where \(p\) is the probability of the positive class, and \(\beta_i\) are model parameters.

It is widely favored for its simplicity, efficiency, and ease of interpretation. Coefficients indicate the direction and strength of feature influence on the likelihood of belonging to a class.

**Pros:**  
- Simple, interpretable classification model.  
- Outputs calibrated probabilities useful for decision-making.  
- Efficient and effective on linearly separable classes.  
- Works well on relatively small or medium-sized datasets.

**Cons:**  
- Assumes a linear decision boundary; limited with complex patterns.  
- Performance degrades if classes are not linearly separable.  
- Requires feature engineering for non-linear or high-dimensional data.

**Use Cases:**  
Common in medical diagnosis, credit scoring, spam detection, and customer churn prediction—any binary classification task requiring explainability and probability estimates.

---

### Random Forest Regression
**Explanation:**  
Random Forest Regression is an ensemble learning method that builds a multitude of decision trees during training and outputs the mean prediction of the individual trees. It can capture complex non-linear relationships and interactions among features without requiring explicit feature engineering.

Each tree is trained on a bootstrap sample of the data and considers a random subset of features at each split, increasing diversity and reducing overfitting compared to a single decision tree.

**Pros:**  
- Handles non-linear and complex relationships well.  
- Robust to noisy data and outliers.  
- Automatically captures feature interactions.  
- Provides feature importance scores.

**Cons:**  
- Computationally intensive and slower to train and predict.  
- Results are less interpretable than linear models; often considered a black box.  
- Requires hyperparameter tuning to achieve optimal performance.

**Use Cases:**  
Appropriate for datasets with complicated patterns and relationships, common in finance, biology, ecology, and any regression setting where the underlying process is not linear.

---

### Ridge Regression
**Explanation:**  
Ridge Regression is a linear regression variant that incorporates L2 regularization to penalize large coefficients. By adding a penalty term proportional to the square of the coefficients, it shrinks coefficients and thus reduces model complexity and collinearity issues. It is especially useful when predictors are highly correlated.

The objective function optimized in Ridge is:  
\[
\min_{\theta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \theta_j^2
\]  
where \(\alpha\) controls the strength of regularization.

**Pros:**  
- Reduces overfitting and improves model generalization.  
- Handles multicollinearity effectively by shrinking coefficients.  
- Keeps all features in the model (does not perform feature selection).

**Cons:**  
- Does not produce sparse models; no automatic feature elimination.  
- Requires tuning of the regularization parameter \(\alpha\).

**Use Cases:**  
Well-suited for datasets with many correlated features or when the number of features is large relative to the number of samples, improving stability and predictive performance of linear models.

---

### Lasso Regression
**Explanation:**  
Lasso Regression adds L1 regularization to linear regression, which adds a penalty equal to the absolute value of the magnitude of coefficients. This encourages sparsity, shrinking some coefficients exactly to zero, thus performing implicit feature selection and simplifying models.

The objective function optimized in Lasso is:  
\[
\min_{\theta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |\theta_j|
\]

**Pros:**  
- Performs regularization and feature selection simultaneously.  
- Produces simpler and more interpretable models.  
- Can improve prediction accuracy by reducing overfitting.

**Cons:**  
- May struggle when features are highly correlated (tending to pick one arbitrarily).  
- Requires careful tuning of \(\alpha\) for best results.

**Use Cases:**  
Effective for high-dimensional datasets with many irrelevant features, common in bioinformatics, text data, and any problem where feature selection is critical.

---

### Support Vector Regression (SVR)
**Explanation:**
Support Vector Regression (SVR) is an extension of Support Vector Machines (SVM) tailored for regression tasks. Instead of classifying data, SVR finds a function that approximates the relationship between input features and continuous target values with an emphasis on creating a margin (called the epsilon-insensitive tube) around the predicted function within which errors are not penalized. The goal is to fit as many data points as possible within this margin, minimizing prediction error without overfitting.

SVR can use different kernel functions (linear, polynomial, RBF) to map input data into higher-dimensional spaces, allowing it to capture complex, nonlinear relationships. The choice of kernel and model hyperparameters like C (regularization strength) and epsilon (width of the margin) controls the balance between model complexity and tolerance for prediction errors.

**Pros:**

Can capture complex nonlinear relationships using kernels.

Robust to outliers through the epsilon margin.

Effective in high-dimensional feature spaces.

**Cons:**

Selecting appropriate kernel and hyperparameters requires care.

Training can be computationally expensive on large datasets.

Less interpretable than simpler linear models.

**Use Cases:**
SVR is well-suited for regression problems with nonlinear patterns, such as predicting stock prices, environmental measurements, or any scenario where relationships are complex but a robust regression model is needed.

---

### Gradient Boosted Regression Trees
**Explanation:**
Gradient Boosted Trees (GBT) are an ensemble learning method that builds a strong predictive model by combining many weak learners, typically shallow decision trees, in a stage-wise fashion. Each successive tree is trained to fit the residual errors (gradients) of the combined prior trees, gradually improving overall prediction accuracy.

GBT models are powerful for capturing nonlinear dependencies and interactions among features since each new tree focuses on correcting the mistakes of the previous ensemble. This sequential learning approach allows for flexibility and strong predictive performance in complex datasets.

**Pros:**

High prediction accuracy, often outperforming other algorithms.

Handles mixed data types and complex feature interactions well.

Can be tuned by adjusting number of trees, learning rate, tree depth for performance optimization.

**Cons:**

Training can be slower compared to simpler models.

Prone to overfitting if not properly tuned.

Less interpretable than single decision trees.

**Use Cases:**
Ideal for regression problems where high accuracy is required on complex, non-linear datasets such as finance, insurance pricing, and many real-world tabular prediction tasks.

---

### Decision Tree Regression
**Explanation:**
Decision Tree Regression is a non-parametric supervised learning method that predicts continuous numerical outputs by recursively partitioning the feature space into regions. The algorithm splits data based on feature thresholds that minimize prediction error within each partition (e.g., mean squared error).

The final prediction for a sample is the average target value of samples within the relevant leaf node. Decision trees are intuitive and easy to visualize, capturing non-linear relationships without needing feature scaling.

**Pros:**

Simple to understand and interpret visually.

Captures nonlinear relationships and feature interactions.

Requires little data preprocessing.

**Cons:**

Prone to overfitting if trees grow too deep.

Sensitive to small variations in data, causing unstable splits.

Typically less accurate than ensemble methods.

**Use Cases:**
Useful as baseline models for regression tasks, exploratory data analysis, and scenarios where model interpretability is crucial, such as basic pricing predictions or preliminary modeling before ensembles.

---

### Gradient Boosted Classification Trees  

**Explanation:**  

Gradient Boosted Classification Trees combine multiple weak learners, typically shallow decision trees, in a sequential manner to build a strong classifier. Each new tree is trained to correct the errors of the combined previous trees by fitting the gradients of the loss function, improving model accuracy iteratively. This boosting approach reduces bias and variance, making it effective for complex classification tasks.  

**Pros:**  
- High classification accuracy and robustness to overfitting if tuned well.  
- Handles heterogeneous data types and complex feature interactions.  
- Flexible loss functions can be used to optimize for different objectives.  

**Cons:**  
- Computationally intensive and slower to train than simpler models.  
- Requires careful hyperparameter tuning to avoid overfitting.  
- Less interpretable compared to single decision trees or linear models.  

**Use Cases:**  

Widely used for classification problems in finance (fraud detection), marketing (customer segmentation), and competitions like Kaggle where model accuracy is critical.

---

### K-Means Clustering  

**Explanation:**  

K-Means is an unsupervised learning algorithm used to partition data into k distinct clusters based on feature similarity. It works iteratively by assigning data points to the nearest cluster centroid and then recalculating centroids until convergence. The objective is to minimize within-cluster variance (sum of squared distances).  

**Pros:**  
- Simple and fast algorithm suitable for large datasets.  
- Easy to implement and interpret clustering results.  
- Scales well to many samples and dimensions.  

**Cons:**  
- Requires specifying the number of clusters (k) in advance.  
- Sensitive to initial centroid placement; may converge to local minima.  
- Assumes clusters are spherical and equally sized.  

**Use Cases:**  

Effective for market segmentation, customer profiling, image compression, and any application needing grouping of unlabeled data.

---

### Polynomial Regression  

**Explanation:**  

Polynomial Regression extends linear regression by modeling the relationship between independent variables and the target as an nth-degree polynomial. This allows capturing nonlinear patterns while still fitting a linear model in the transformed polynomial feature space. The model equation is:  
$$
\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_n x^n
  

**Pros:**  
- Captures nonlinear relationships without complex models.  
- Simple extension of linear regression with interpretable coefficients.  
- Flexible in modeling curves by adjusting polynomial degree.
  
**Cons:**  
- Prone to overfitting if polynomial degree is too high.  
- Can become numerically unstable for very high degrees.  
- Interpretability decreases as degree increases.  
**Use Cases:**
  
Useful in scenarios where relationships are polynomial but data is limited, such as modeling growth curves, time series trends, and physics-based processes.

---

### Naive Bayes  

**Explanation:**  
Naive Bayes classifiers are probabilistic models that apply Bayes’ theorem with a strong assumption of feature independence given the class label. Common variants include Gaussian, Multinomial, and Bernoulli Naive Bayes. Despite the “naive” assumption, the models are effective and computationally efficient for many classification tasks, especially with high-dimensional data such as text.  

**Pros:**  
- Fast, simple, and scalable to large datasets.  
- Performs well with high-dimensional feature spaces.  
- Robust to irrelevant features due to independence assumption.  

**Cons:**  
- Assumes feature independence, which is rarely true in practice.  
- Can perform poorly if features are highly correlated.  
- Less effective with continuous data unless using Gaussian variant.  

**Use Cases:**  
Widely applied in text classification (spam detection, sentiment analysis), document categorization, and real-time prediction applications requiring quick decisions.

---

# K-Nearest Neighbors (KNN)

## KNN for Classification

**Explanation:**
K-Nearest Neighbors (KNN) for classification is a simple, non-parametric, and instance-based learning algorithm. It classifies a data point based on the majority vote of its *k* nearest neighbors in the feature space. The distance metric (commonly Euclidean, Manhattan, or Minkowski) determines which points are considered "closest."  

Since KNN doesn’t explicitly learn model parameters, it can adapt well to complex decision boundaries, but its performance depends heavily on the choice of *k* and the distance metric. A smaller *k* may lead to overfitting, while a larger *k* can smooth decision boundaries but may underfit.

**Pros:**
- Easy to implement and understand.
- No assumption about data distribution.
- Naturally handles multi-class classification.
- Effective for small datasets with clear separation.

**Cons:**
- Computationally expensive during inference, as it requires distance calculation for all points.
- Performance degrades with high-dimensional data (curse of dimensionality).
- Sensitive to irrelevant features and data scaling.
- Requires careful choice of *k* to balance bias and variance.

**Use Cases:**
- Image recognition and classification.
- Recommender systems (finding similar users/items).
- Medical diagnosis and classification tasks.
- Text classification such as spam detection.


---

## KNN for Regression

**Explanation:**
K-Nearest Neighbors can also be used for regression tasks by predicting the value of a new data point as the average (or weighted average) of the values of its *k* nearest neighbors. Instead of voting for a class, neighbors contribute their actual numerical values to the prediction.  

Weighted KNN regression often assigns higher influence to closer neighbors, improving accuracy in datasets with local variations. However, as with classification, the algorithm is computationally intensive at inference and sensitive to noise.

**Pros:**
- Simple and intuitive for regression problems.
- Naturally handles nonlinear relationships between features and output.
- Weighted versions can improve performance by prioritizing closer neighbors.

**Cons:**
- Computationally heavy with large datasets.
- Sensitive to noisy data points, which can skew averages.
- Suffers from the curse of dimensionality.
- Requires tuning of *k* and distance metric for optimal performance.

**Use Cases:**
- Predicting housing prices based on location and features.
- Forecasting demand or sales in retail.
- Predicting patient health metrics from medical records.
- Environmental modeling (e.g., predicting pollution levels).


---

## Getting Started

### Prerequisites

Make sure you have Python installed (version 3.7 or above).

Install the required dependencies:

```
pip install -r requirements.txt
```

### Running the Linear Regression example

You can run the Linear Regression code from the Jupyter notebook in the `notebooks/` folder or use the Python scripts in `src/`.

---

## Future Work

Plans include adding implementations for more ML algorithms like:

- Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines
- K-Nearest Neighbors
- Neural Networks (Deep Learning)
- Clustering techniques (K-Means, DBSCAN, etc.)

New algorithms will also include visualization, thorough evaluation, and practical insights.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests with new algorithms, improvements, or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or collaboration, please reach out on GitHub or through my profile.

---

Thank you for visiting my AI-ML-Algorithms repository!

---

"The real risk of AI isn’t malice, but competence. A highly intelligent AI can achieve its goals in ways we never intended."

— Stuart Russell
