
````markdown
# MNIST MLP Experiments (Scikit-Learn)

This project trains multiple Multi-Layer Perceptron (MLP) neural networks on the MNIST handwritten digit dataset in CSV format. The goal is to compare different activation functions, optimizers, learning rates, and hidden-layer sizes to understand how each affects model performance and training behavior.

The code is written in Python using `scikit-learn`, and supports file upload through Google Colab.

---
## 📁 Dataset

This project uses the **MNIST-in-CSV** dataset from Kaggle:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/versions/2

This version of MNIST was converted to CSV format by Joseph Redmon to make it easier for beginners to load without needing IDX parsing.

Files included:

- `mnist_train.csv` — 60,000 training examples  
- `mnist_test.csv` — 10,000 test examples  

Each row contains:

- `label` (0–9)
- `784 pixel columns` named in grid format (`1x1`, `1x2`, … `28x28`)

---

## 🚀 What the Script Does

### 1. Load the dataset  
The notebook accepts uploaded files in Google Colab:

```python
train = pd.read_csv("mnist_train.csv")
test  = pd.read_csv("mnist_test.csv")
````

### 2. Preprocess the data

* Training split: first 5000 samples
* Test split: next 5000 samples
* Normalize pixel values to `[0, 1]`

### 3. Train and test multiple MLP configurations

The script evaluates several neural network setups:

### **🔹 Adam + ReLU (baseline)**

* Hidden units: 100
* Learning rate: 0.1
* max_iter: 20

### **🔹 SGD + ReLU**

* High learning rate (0.1)
* Shows divergence and unstable loss

### **🔹 Adam + Identity Activation**

* Reduced performance (linear model)
* Demonstrates value of nonlinear activations

### **🔹 LBFGS + ReLU**

* Full-batch optimizer
* Does not print iteration loss in sklearn

### **🔹 Adam + ReLU (200 hidden units)**

* Shows improvement from increasing model capacity

---

## 📊 Results Summary

| Configuration           | Notes                                  |
| ----------------------- | -------------------------------------- |
| Adam + ReLU             | Best stability and accuracy            |
| SGD + ReLU              | Unstable with lr=0.1, high loss        |
| Adam + Identity         | Linear model → bad accuracy            |
| LBFGS + ReLU            | Converges silently, no verbose logging |
| Adam + ReLU (200 units) | Better accuracy due to larger network  |

---

## 🧠 Conceptual Questions & Answers

### **1. Why is the loss much greater when using SGD vs Adam?**

SGD uses a fixed learning rate and does not adapt to gradient magnitude.
With a high learning rate (0.1) and unscaled inputs, SGD diverges.
Adam adapts updates per-weight → stable learning.

### **2. Does learning rate mean the interval the MLP should try?**

Yes — learning rate = step size in weight space.
Too large → divergence, too small → slow learning.

### **3. Why does ReLU give more consistent loss than identity?**

Identity activation is linear, so the whole network becomes a linear model.
MNIST is not linearly separable → network plateaus and shows unstable loss.

### **4. Why does LBFGS show no loss even with verbose on?**

LBFGS is a full-batch quasi-Newton optimizer.
scikit-learn does not print iteration loss for LBFGS.

### **5. Is the printed loss mean absolute error?**

No — `MLPClassifier` uses **cross-entropy (log-loss)** for classification.

### **6. Why does the MLP show a big loss drop on iteration 2 and then flatten?**

Adam makes a large first step as it adapts the gradient, then refines weights with smaller steps → smaller changes per iteration.

---

## 🛠 Requirements

```
pandas
scikit-learn
numpy
google-colab (optional)
```

---

## ▶️ Running the Project

Upload the MNIST CSV files and run the script:

```python
python mnist_mlp_experiments.py
```

In Google Colab, the upload widget appears automatically.

