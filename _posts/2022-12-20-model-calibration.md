---
layout: post
usemathjax: true
title: "Model Calibration"
subtitle: "Understanding and Improving the Accuracy of Your Predictive Models through Calibration"
date: 2022-12-26 23:45:13 -0400
background: '/img/posts/calibration/model-calibration.jpeg'
---

<h1 style="text-align: center;">Intro</h1>

Although sometimes disregarded, calibration is a crucial component of building machine learning classifiers. It provides information on the uncertainty of the model, which can then be shared with end users or used to process the model outputs further. The theory and practice of calibrating models to extract more value from their predictions will be covered in this blog.  
  
Model calibration is a method used to improve the reliability of machine learning models by adjusting the predicted probabilities of a model to be more closely aligned with the true probabilities of the outcomes. This is typically done by fitting a separate model to the predicted probabilities of the original model.  
  
The goal of model calibration is to improve the accuracy of probability estimates, which can be especially important when using a model for decision making or risk assessment. For example, if a model is predicting the probability of an event occurring, it is important that the predicted probabilities accurately reflect the true likelihood of the event occurring.  
  
Assume you have two observations and a binary classifier; the model assigns the two observations scores of 0.7 and 0.9, respectively. Does the sample with the 0.9 score have a greater likelihood of falling into the positive category? This is accurate for some models, but it might not be for others.

<h2 style="text-align: center;">Calibrated Model</h2>

If a model that can identify if an image contains a cat or not is given a collection of 10 pictures and produces a probability of 0.6 for each picture that contains a cat, we can anticipate that the set will contain 6 cat pictures.  
  
<h2 style="text-align: center;">Non Calibrated Model  </h2>

Let's use a birth gender prediction as an example (male in our case). One of the models consistently produces 0.5 (or 50%), whereas the other alternates between predicting 0.3 and 0.7 and consistently produces a different outcome. The first of these two models is calibrated precisely, whilst the other is not. The initial model's response that 50% of births were males was accurate, which is why that is the case. However, of the births for which the second model provided a response of 30%, 50% were male, and the same was true for responses of 70%.  
  
To summarize, probabilities are calibrated so that a class forecast made with confidence $$P$$ is accurate 100*$$P$$% of the time.  
  
| Algorithms | Well-Calibrated | Reason |  
|---|---|---|  
| Logistic regression | Yes | Logistic regression models output probabilities that are well-calibrated by design. This is because the logistic function, which is used to map the predicted scores to probabilities, is a sigmoidal function that maps any value to a value between 0 and 1.|  
| Naive Bayes | Yes\No | Naive Bayes models are based on the assumption that features are independent and equally important, which allows them to output well-calibrated probabilities. However, this assumption is often not met in practice, which can lead to poorly calibrated probabilities. |  
| Decision tree | No | Decision trees do not output probabilities, so they are not well-calibrated by default. Instead, they output a class label for each example. |  
| Random Forests | Yes | Random forests are an ensemble model made up of multiple decision trees. By averaging the predictions of the individual trees, random forests can output well-calibrated probabilities. |  
| Support vector machine | No | Support vector machines do not output probabilities, so they are not well-calibrated by default. Instead, they output a class label for each example based on the decision boundary learned during training. |  
| Neural network | No | Neural networks do not output probabilities, so they are not well-calibrated by default. Instead, they output a score for each class, which can be transformed into a probability using a function like the softmax function. However, the predicted probabilities may not be well-calibrated if the neural network is poorly designed or trained or have high depth using a high number of hidden layers. | 

<br/><br/>
<h2 style="text-align: center;">Calibration Plot</h2>
  
A calibration plot is a graphical representation of the performance of a model in predicting a target variable. It is used to assess the accuracy of a model's predictions and to identify any potential bias in the model's predictions.  
  
To create a calibration plot, the predicted values from a model are plotted on the x-axis and the corresponding observed values are plotted on the y-axis. A diagonal line is also plotted on the graph, representing perfect agreement between the predicted and observed values. Points that fall on this line indicate that the model's predictions are accurate. Points that fall above or below the line indicate that the model's predictions are either overestimating or underestimating the true values, respectively.  
  
Calibration plots can be useful for identifying any potential bias in a model's predictions. For example, if the points on the plot are consistently above the diagonal line, this may indicate that the model is consistently overestimating the true values. On the other hand, if the points are consistently below the diagonal line, this may indicate that the model is consistently underestimating the true values.  

<details>
  <summary><b>More on calibration curve</b></summary>

	<p class="tab">Calibration curve often called reliability curve.</p>

	<p class="tab">Reliability curves are a graphical representation of the relationship between the predicted probabilities of an event occurring and the actual observed frequency of that event. They are commonly used in model calibration to assess the accuracy of a model's predictions.</p>

	<p class="tab">The x-axis of a reliability curve represents the predicted probability of an event occurring, while the y-axis represents the observed frequency of that event. The ideal reliability curve would be a straight line with a slope of 1 and an intercept of 0, indicating that the model's predictions are perfectly calibrated to the observed data.</p>

	<p class="tab">However, in practice, most models will deviate from this ideal line, often resulting in a curved shape. A model that is over-calibrated will have a curve that is above the ideal line, indicating that the model is predicting more events than are actually occurring. Conversely, a model that is under-calibrated will have a curve that is below the ideal line, indicating that the model is predicting fewer events than are actually occurring.</p>

	<p class="tab">The distance between the reliability curve and the ideal line is known as the calibration error, and it is used as a measure of the model's overall accuracy. A model with a low calibration error is considered to be well-calibrated and is more likely to produce accurate predictions.</p>

	<p class="tab">In summary, reliability curves are a useful tool for assessing the accuracy of a model's predictions by comparing the predicted probabilities of an event occurring to the actual observed frequency of that event. A well-calibrated modl will have a reliability curve that closely aligns with the ideal line, indicating that its predictions are accurate.</p>
</details>
To demonstrate this kind of plots you can use the following code:  

```python
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  

%matplotlib inline  

SEED = 42
X, y = make_classification(n_samples=10_000, 
                           n_classes=2,
                           n_features=20, 
                           n_informative=2, 
                           n_redundant=10, 
                           random_state=SEED)
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=SEED)

lr = LogisticRegression(C=1.0)
gnb = GaussianNB()
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

clf_list = [
(lr,  "Logistic"),
(gnb,  "Naive Bayes"),
(gnb_isotonic,  "Naive Bayes + Isotonic"),
(gnb_sigmoid,  "Naive Bayes + Sigmoid"),
]

fig = plt.figure(figsize=(10,  10))
gs = GridSpec(4,  2)
colors = plt.cm.get_cmap("Dark2")
ax_calibration_curve = fig.add_subplot(gs[:2,  :2])
calibration_displays = {}

for i,  (clf, name)  in  enumerate(clf_list):
	clf.fit(X_train, y_train)
	display = CalibrationDisplay.from_estimator(
	clf,
	X_test,
	y_test,
	n_bins=10,
	name=name,
	ax=ax_calibration_curve,
	color=colors(i),
	)

	calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

grid_positions = [(2,  0),  (2,  1),  (3,  0),  (3,  1)]

for i,  (_, name)  in  enumerate(clf_list):
	row, col = grid_positions[i]

	ax = fig.add_subplot(gs[row, col])
	ax.hist(
	calibration_displays[name].y_prob,
	range=(0,  1),
	bins=10,
	label=name,
	color=colors(i),
	)
	ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
```

![add image](/img/posts/calibration/calibration_plot.png)
[Source](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py)

<h1 style="text-align: center;">Calibration Methods</h1>

<h2 style="text-align: center;">Platt scaling</h2>
This method involves fitting a logistic regression model to the predicted probabilities of the original model.  

Platt scaling, also known as Platt calibration, is a method used to convert the predicted probabilities output by a classification model into well-calibrated probabilities. In other words, Platt scaling is used to make sure that the predicted probabilities of a model are accurate, or "calibrated", with respect to the true probabilities of the classifier classes.

The basic idea behind Platt scaling is to use a logistic function to map the predicted probabilities to the true probabilities. The logistic function is given by the following formula:

$$P(y=1 \mid x)=\frac{1}{1+\exp -(A f(x)+B)}$$

The parameters A and B in the equation are scaling parameters that control how the scaling is applied. These parameters are determined through a process called maximum likelihood estimation, which finds the values of A and B that minimize the difference between the predicted probabilities and the true probabilities in the data.

In other words, maximum likelihood estimation is an algorithm that is used to find the optimal values of A and B that best map the predicted probabilities to the true probabilities within each bin or neighborhood of data.

Note that Platt scaling only works well when the classifier is well-calibrated to begin with. If the classifier is poorly calibrated, Platt scaling may not improve the calibration significantly.

<h3 style="text-align: center;">Implementation</h3>

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

X, y = make_classification(n_samples=10_000, n_classes=2, n_features=2, n_redundant=0, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=SEED)

base_clf = GaussianNB()
calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method='sigmoid')
calibrated_clf.fit(X_train, y_train)
calibrated_clf.predict(X_test)
```

Note that we are using the sane X_train and y_train in our calibrated model to avoid overfitting (this is way we are using cross validation).

<h2 style="text-align: center;">Isotonic regression</h2>
This method fits a non-parametric model that is constrained to be monotonically increasing.  
  
Isotonic regression is a method used to calibrate the output of a classifier so that it can be interpreted as a probability. It is a non-parametric method, which means that it does not make any assumptions about the functional form of the relationship between the predicted probabilities and the true probabilities.  

The main difference between linear regression and isotonic regression is that linear regression models the relationship between the input and output variables as a straight line, while isotonic regression models the relationship as a step function with constant slopes within each interval. Isotonic regression is useful in situations where the relationship between the input and output variables is known to be monotonic but may not be linear.

The regression is trying to solve:

$$\min _f \sum_{i=1}^n w_i\left(f\left(x_i\right)-y_i\right)^2$$

Example of f(x):

$$f(x)= \begin{cases}\hat{y}_1 & \text { if } x \leq x_1 \\ \hat{y}_i+\frac{x-x_i}{x_{i+1}-x_i}\left(\hat{y}_{i+1}-\hat{y}_i\right) & \text { if } x_i \leq x \leq x_{i+1} \\ \hat{y}_n & \text { if } x \geq x_n\end{cases}$$

Remember that isotonic regression is a popular method for calibrating classifiers because it is simple to implement and does not require any assumption about the functional form of the relationship between the predicted probabilities and the true probabilities. However, it can be sensitive to noise in the data and may not perform well when the classifier is poorly calibrated to begin with.  
  
<h3 style="text-align: center;">Implementation</h3>

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

X, y = make_classification(n_samples=10_000, n_classes=2, n_features=2, n_redundant=0, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=SEED)

base_clf = GaussianNB()
calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method='isotonic')
calibrated_clf.fit(X_train, y_train)
calibrated_clf.predict(X_test)
```

Note that we are using the sane X_train and y_train in our calibrated model to avoid overfitting (this is way we are using cross validation).

<h4 style="text-align: center;">The Difference Between Isotonic Regression And Platt Scaling</h4>
Isotonic regression and Platt scaling are both methods for calibrating the predicted probabilities of a classifier. They both transform the model's output into well-calibrated probabilities that reflect the true likelihood of an example belonging to a certain class.

The main difference between the two methods is the shape of the calibration curve that they use. Platt scaling uses a sigmoid curve, which implies that the probability distribution may be distorted in a sigmoid shape. Isotonic regression, on the other hand, projects the predicted probabilities onto a set of increasing functions (monotonic functions) in order to correct for any monotonic distortion in the probability distribution. This makes isotonic regression a more powerful calibration method that is able to correct for a wider range of distortions.

However, isotonic regression can be prone to overfitting if the dataset is small, so it is not always the best choice. 

<h2 style="text-align: center;">Temperature scaling </h2>
Temperature scaling is a method used to calibrate the predicted probabilities output by a classification model. The basic idea behind temperature scaling is to apply a non-linear transformation to the predicted probabilities using a parameter called the temperature.

The temperature is a hyperparameter that controls the "sharpness" of the transformation. A high temperature results in a smooth transformation, while a low temperature results in a sharp transformation.

The temperature scaling transformation is given by the following formula:

$$P\left(y_i \mid x\right)=\frac{e^{\operatorname{logits}_{i / T}}}{\sum_j e^{\operatorname{logits}_j / T}}$$

where p' is the calibrated probability, p is the predicted probability and T is the temperature.

To use temperature scaling to calibrate a model, we need to choose a temperature value that produces well-calibrated probabilities. This can be done through cross-validation, where we try different temperature values and evaluate the performance of the calibrated model on a validation set.

Here is a step-by-step explanation of the temperature scaling method:  
  
1. Train a classifier on your dataset. The classifier should output a score for each class (e.g., a score between 0 and 1 for each class in a binary classification task).  
  
2. Compute the predicted probabilities for each example in your dataset. You can do this by applying the softmax function to the scores output by the classifier. The softmax function maps the scores to a probability distribution over the classes.  
  
3. Choose a temperature hyperparameter for the temperature scaling. The temperature is a scalar value that determines how "sharp" or "diffuse" the predicted probabilities are. A high temperature results in more diffuse probabilities, while a low temperature results in sharper probabilities.  
  
4. Scale the predicted probabilities using the temperature hyperparameter. You can do this by dividing the predicted probabilities by the temperature.  
  
5. Use the scaled probabilities to make predictions. You can now use the scaled probabilities to make predictions about the class labels of new examples.
  
  
Note that temperature scaling is a simple and effective method for improving the calibration of a classifier. It is particularly useful when the classifier is overconfident, meaning that it outputs very high or very low probabilities for certain classes. By scaling the probabilities with a temperature hyperparameter, you can adjust the confidence of the classifier to be more reasonable. However, temperature scaling may not be effective if the classifier is poorly calibrated to begin with.  
  
<h2 style="text-align: center;">Conclusions </h2>

In summary, model calibration is a crucial step in the machine learning process that ensures that the predicted probabilities of a model are accurate and reliable. Calibration can be performed using a variety of methods, including Platt scaling, Isotonic regression and temperature scaling. By calibrating a model, we can improve its performance and increase its reliability for use in real-world applications. It is important to keep in mind that model calibration should be performed as part of the model development process, and not just before deploying the model in production. Properly calibrated models can provide more accurate and trustworthy predictions, which can lead to better decision-making and improved outcomes.

