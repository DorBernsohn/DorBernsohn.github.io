---
layout: post
usemathjax: true
title: "Correlation"
subtitle: "A Comprehensive Guide to Different Types of Correlation and Their Implications"
date: 2023-01-26 23:45:13 -0400
background: '/img/posts/correlation/correlation.jpeg'
---

<h1 style="text-align: center;">Intro</h1>
As a data scientist, correlation is a fundamental concept that is used to understand the relationship between two or more variables. Correlation analysis is a statistical method that measures the strength of the relationship between two variables and the direction of this relationship. It is an important tool for exploring relationships between variables in large datasets and helps data scientists to identify trends, patterns, and relationships that may be hidden within the data.

There are many different methods for calculating correlation, each with its own strengths and weaknesses. In this review, we will take a closer look at some of the most commonly used correlation methods, including Pearson's Correlation, Spearman's Correlation, and Kendall's Tau Correlation. We will discuss the mathematical foundations of each method, its strengths and weaknesses, and the appropriate use cases for each method.

<h2 style="text-align: center;">Table Of Content</h2>

[Pearson's Correlation Coefficient](#correlation1)

[Spearman's Rank Correlation Coefficient](#correlation2)

[Kendall's Tau Rank Correlation Coefficient](#correlation3)

[Point-Biserial Correlation Coefficient](#correlation4)

[Phi Coefficient - Matthews correlation coefficient (MCC)](#correlation5)

[Goodman and Kruskal's Gamma](#correlation6)

[Cramers V](#correlation7)

[Theil's U statistic (Uncertainty coefficient)](#correlation8)


<h2 style="text-align: center;" id="correlation1">Pearson's Correlation Coefficient</h2>

Pearson's Correlation Coefficient is a measure of the linear association between two continuous variables. It measures the strength and direction of a linear relationship between two variables.

A data scientist might use Pearson's Correlation Coefficient to determine whether a relationship exists between two variables and, if so, the nature of that relationship. For example, one might want to determine if there is a relationship between the height and weight of a population.

The Pearson's Correlation Coefficient ranges from -1 to 1, where values close to 1 indicate a strong positive linear relationship between the variables, values close to -1 indicate a strong negative linear relationship, and values close to 0 indicate no linear relationship.

It's important to note that Pearson's Correlation Coefficient assumes that the relationship between the variables is linear. If the relationship is not linear, the coefficient may not accurately reflect the true relationship between the variables. In such cases, other correlation measures, such as Spearman's Rank Correlation Coefficient, should be used.

Formula:

$$
r_{x y}=\frac{\sum_{i=1}^n\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sqrt{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2} \sqrt{\sum_{i=1}^n\left(u_i-\bar{y}\right)^2}}
$$

Where:

- $$n$$ is sample size

- $$x_{i},y_{i}$$ are the individual sample points indexed with $$i$$

- $${\textstyle {\bar {x}}={\frac {1}{n}}\sum _{i=1}^{n}x_{i}}$$ (the sample mean); and analogously for $${\bar {y}}$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

method_name = "Pearson's Correlation Coefficient"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

# Generate linear data
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

# Calculate Pearson's Correlation Coefficient for linear data
correlation, p = stats.pearsonr(x, y)

# Plot linear data
axes[0].scatter(x, y)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f"Linear Data Correlation: {correlation:.2f}")

# Generate non-linear data
x = np.random.normal(0, 1, 100)
y = x**2 + np.random.normal(0, 0.5, 100)

# Calculate Pearson's Correlation Coefficient for non-linear data
correlation, p = stats.pearsonr(x, y)

# Plot non-linear data
axes[1].scatter(x, y)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(f"Non-Linear Data Correlation: {correlation:.2f}")

fig.suptitle(method_name, fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
```
![add image](/img/posts/correlation/Pearsons Correlation Coefficient.png)

<h2 style="text-align: center;" id="correlation2">Spearman's Rank Correlation Coefficient</h2>

Spearman's Rank Correlation Coefficient is a measure of the monotonic association between two continuous or ordinal variables. It measures the strength and direction of a monotonic relationship between two variables.

A data scientist might use Spearman's Rank Correlation Coefficient when they want to determine if there is a relationship between two variables, but the relationship is not necessarily linear. For example, One might want to determine if there is a relationship between IQ scores and test performance, where a high IQ score might result in higher test performance, but the relationship between the two variables may not be linear.

Spearman's Rank Correlation Coefficient is calculated based on the rankings of the two variables, rather than their actual values. This makes it well-suited for data that may not have a linear relationship, but still has a monotonic relationship (i.e. the variables increase or decrease together, but not necessarily at a constant rate).

The Spearman's Rank Correlation Coefficient ranges from -1 to 1, where values close to 1 indicate a strong positive monotonic relationship between the variables, values close to -1 indicate a strong negative monotonic relationship, and values close to 0 indicate no monotonic relationship.

It is a useful alternative to Pearson's Correlation Coefficient when the relationship between the variables is not linear.

Formula:

$$\rho=1-\frac{6 \sum d_{i}^{2}}{n (n^{2}-1)}$$

Where:

- $$\rho$$ 	=	Spearman's rank correlation coefficient

- $$d_{i}$$	=	difference between the two ranks of each observation

- $$n$$	    =	number of observations

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

method_name = "Spearman's Rank Correlation Coefficient"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

# Generate linear data
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

# Calculate Spearman's Rank Correlation Coefficient for linear data
correlation, p = stats.spearmanr(x, y)

# Plot linear data
axes[0].scatter(x, y)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f"Linear Data Correlation: {correlation:.2f}")

# Generate non-linear data
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = x**2 + np.random.normal(0, 0.5, 100)

# Calculate Spearman's Rank Correlation Coefficient for non-linear data
correlation, p = stats.spearmanr(x, y)

# Plot non-linear data
axes[1].scatter(x, y)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(f"Non-Linear Data Correlation: {correlation:.2f}")

fig.suptitle(method_name, fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
```
![add image](/img/posts/correlation/Spearmans Rank Correlation Coefficient.png)

<h2 style="text-align: center;" id="correlation3">Kendall's Tau Rank Correlation Coefficient</h2>

The Kendall's Tau Rank Correlation Coefficient is a non-parametric statistical measure of the strength and direction of the association between two variables. Unlike the Pearson's correlation coefficient, which measures linear relationships, the Kendall's Tau rank correlation measures monotonic relationships, meaning that as one variable increases, the other variable either increases or decreases in a consistent manner, without necessarily being a straight line.

Kendall's Tau measures the number of concordant and discordant pairs in the ranked data. A concordant pair is defined as two pairs (x1, y1) and (x2, y2) such that x1 < x2 and y1 < y2 or x1 > x2 and y1 > y2. Discordant pairs are defined as pairs such that x1 < x2 and y1 > y2 or x1 > x2 and y1 < y2. Kendall's Tau is calculated as the ratio of the number of concordant pairs to the total number of pairs, minus the ratio of discordant pairs to the total number of pairs.

Kendall's Tau is a robust measure of correlation and is less sensitive to outliers than Pearson's correlation. It is also a good choice for datasets with non-normal distributions and can handle tied ranks. However, it is less sensitive to the strength of the relationship and may not be as effective as Pearson's correlation in detecting linear relationships.

Formula:

$$
\tau=\frac{\text { (number of concordant pairs) }-\text { (number of discordant pairs) }}{\text { (number of pairs) }}=1-\frac{\text { 2(number of discordant pairs) }}{\left(\begin{array}{l}
n \\
2
\end{array}\right)}
$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

method_name = "Kendall's Tau Rank Correlation Coefficient"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

# Generate linear data
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

# Calculate Kendall's Tau Rank Correlation Coefficient for linear data
correlation, p = stats.kendalltau(x, y)

# Plot linear data
axes[0].scatter(x, y)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f"Linear Data Correlation: {correlation:.2f}")

# Generate non-linear data
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = x**2 + np.random.normal(0, 0.5, 100)

# Calculate Kendall's Tau Rank Correlation Coefficient for non-linear data
correlation, p = stats.kendalltau(x, y)

# Plot non-linear data
axes[1].scatter(x, y)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(f"Non-Linear Data Correlation: {correlation:.2f}")

fig.suptitle(method_name, fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
```
![add image](/img/posts/correlation/Kendalls Tau Rank Correlation Coefficient.png)

<h2 style="text-align: center;" id="correlation4">Point-Biserial Correlation Coefficient</h2>

The Point-Biserial Correlation Coefficient is a type of correlation coefficient used to measure the association between a binary variable and a continuous variable. The binary variable is typically represented as a dichotomous variable, with two levels (e.g. yes/no, male/female). The continuous variable can be any numerical variable.

The Point-Biserial Correlation Coefficient measures the difference between the means of the continuous variable for each level of the binary variable. If the means of the continuous variable are significantly different between the two levels of the binary variable, then there is a positive or negative association between the two variables, depending on the direction of the difference.

Formula:

$$
r_{p b}=\frac{M_1-M_0}{s_n} \sqrt{\frac{m_1 n_0}{n^2}}
$$

Where:
- $$s_n$$ is the standard deviation.
- $$M_1$$ being the mean value on the continuous variable $$X$$ for all data points in group 1, and $$M_0$$ the mean value on the continuous variable $$X$$ for all data points in group 2.
- $$n_1$$ is the number of data points in group $$1$$, $$n_0$$ is the number of data points in group $$2$$ and n is the total sample size.

```python
import scipy.stats as stats
data = np.array([[4, 2, -2, 1, 8, -6, -9, -8, 11, 13, 12, -14],
        [1, 2, 2, 4, 9, 8, 9, 6, 14, 12, 13, 12],
        [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]]).transpose()
df = pd.DataFrame(data)
df = df.rename(columns={0: 'x', 1: 'y', 2: 'label'})

x_point_biserial, x_p_value = stats.pointbiserialr((df['label'] == 1), df['x'])
y_point_biserial, y_p_value = stats.pointbiserialr((df['label'] == 1), df['y'])

res_df = pd.DataFrame({'Variable': ['x', 'y'], 'Correlation': [x_point_biserial, y_point_biserial], 'P-Value': [x_p_value, y_p_value]})
res_df['Rejected Null-Hypothesis'] = res_df['P-Value'].apply(lambda x: x < 0.05)

method_name = "Point-Biserial Correlation Coefficient"
fig, ax = plt.subplots()

colors = {0:'red', 1:'blue'}


ax.scatter(df['x'], df['y'], c=df['label'].map(colors))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f"X Correlation: {x_point_biserial:.2f}, Y Correlation: {y_point_biserial:.2f}")
fig.suptitle(method_name, fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.8)
```

| Variable | Correlation | P-Value | Rejected Null-Hypothesis |
| --- | --- | --- | --- |
| x | 0\.8646 | 0\.0003 | true |
| y | 0\.0510 | 0\.8750 | false |

![add image](/img/posts/correlation/Point-Biserial Correlation Coefficient.png)

<h2 style="text-align: center;" id="correlation5">Phi Coefficient - Matthews correlation coefficient (MCC)</h2>

Both the Phi Coefficient and the Matthews Correlation Coefficient (MCC) are measures of association between two binary variables. They are used to evaluate the performance of a binary classification model by comparing its predicted labels to the true labels. The Phi Coefficient is a correlation coefficient that measures the degree of association between two binary variables, usually represented as a 2x2 contingency table. The Matthews Correlation Coefficient (MCC) is another measure of association between two binary variables, but it is considered to be a more balanced measure than the Phi Coefficient, especially when the data is imbalanced. 

The Phi Coefficient is calculated as the ratio of the number of cases where both binary variables are present to the total number of cases, minus the ratio of cases where both binary variables are absent to the total number of cases. If the number of cases where both binary variables are present is higher than expected by chance, then the Phi Coefficient is positive and indicates a positive association between the two binary variables. If the number of cases where both binary variables are absent is higher than expected by chance, then the Phi Coefficient is negative and indicates a negative association between the two binary variables.

The Phi Coefficient is a useful measure of association for binary variables, especially in situations where the sample size is small. It is less sensitive to outliers and does not require a normal distribution of the data.

Formula:

$$
\phi=\frac{n_{11} n_{00}-n_{10} n_{01}}{\sqrt{n_{1\odot} n_{0\odot} n_{\odot1} n_{\odot1}}} .
$$


Where:

- $$n11$$, $$n10$$, $$n01$$, $$n00$$, are non-negative counts of numbers of observations that sum to $$n$$, the total number of observations.

Another use case for this correlation is when we have imbalanced binary classification and we want to measure the accuracy by calculating the:

$$
\mathrm{MCC}=\frac{T P \times T N-F P \times F N}{\sqrt{(T P+F P)(T P+F N)(T N+F P)(T N+F N)}}
$$

- A score of -1 denotes complete disagreement between predicted and actual classes.
- 0 is equivalent to making a completely arbitrary guess.
- A score of 1 denotes total agreement between predicted and actual classes.

```python
from sklearn.metrics import matthews_corrcoef
#define array of actual classes
actual = np.repeat([1, 0], repeats=[20, 380])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[15, 5, 5, 375])

#calculate Matthews correlation coefficient
matthews_corrcoef(actual, pred)
```
[Source](https://www.statology.org/matthews-correlation-coefficient-python/#:~:text=model%20in%20Python%3A-,import%20numpy%20as%20np,-from%20sklearn.)

<h2 style="text-align: center;" id="correlation6">Goodman and Kruskal's Gamma</h2>

Goodman and Kruskal's Gamma (also known as Gamma) is a rank-based correlation coefficient that measures the association between two ordinal variables. Ordinal variables are variables that can be ordered or ranked, but the distances between the levels are not necessarily equal. For example, education levels (e.g. high school, bachelor's degree, master's degree) are an example of ordinal variables.

Gamma measures the strength and direction of the association between two ordinal variables by calculating the ratio of concordant to discordant pairs of observations. A concordant pair is a pair of observations where the rank of one variable is higher when the rank of the other variable is also higher, and the rank of one variable is lower when the rank of the other variable is also lower. A discordant pair is a pair of observations where the rank of one variable is higher when the rank of the other variable is lower, or vice versa. The Gamma coefficient ranges from -1 to 1, with a value of 1 indicating a perfect positive association between the two variables, a value of -1 indicating a perfect negative association, and a value of 0 indicating no association.

Goodman and Kruskal's Gamma is commonly used in ordinal data analysis, such as when studying the association between two categorical variables with ordered categories. It is a useful measure of association when the data does not meet the assumptions of other correlation measures, such as when the data is not normally distributed or when the sample size is small.

$$
G=\frac{N_s-N_d}{N_s+N_d}
$$

Where:
- $$N_s$$, the number of pairs of cases ranked in the same order on both variables (number of concordant pairs).
- $$N_d$$, the number of pairs of cases ranked in reversed order on both variables (number of reversed pairs).

<h2 style="text-align: center;">Partial Correlation</h2>

Partial correlation is a statistical method used to measure the strength of the relationship between two variables, while adjusting for the influence of one or more additional variables. The partial correlation coefficient represents the correlation between two variables after removing the effects of the other variables that are being adjusted for.

Partial correlation is useful when analyzing complex relationships between multiple variables, where it is important to disentangle the specific relationship between two variables from the influences of other variables. It is often used in fields such as psychology, epidemiology, and economics, where multiple confounding factors need to be considered in order to interpret the relationships between variables accurately.

```python
import numpy as np
from scipy import stats, linalg

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr

data = np.array([[4, 2, 2, 1, 8, 6, 9, 8, 11, 13, 12, 14],
        [1, 2, 2, 4, 9, 8, 9, 6, 14, 12, 13, 12],
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]).transpose()
partial_corr(data)
array([[1.        , 0.93699911, 0.19315731],
       [0.93699911, 1.        , 0.02766727],
       [0.19315731, 0.02766727, 1.        ]])
```
[Helpful Thread](https://gist.github.com/fabianp/9396204419c7b638d38f)

<h2 style="text-align: center;">Kremer's Rho (Kendall's Tau concordance correlation)</h2>


Kremer's Rho, also known as Kendall's Tau concordance correlation, is a non-parametric statistical measure used to evaluate the correlation between two sets of continuous or ordinal data. It is used when the assumptions of linearity and normality made by traditional methods, such as Pearson's correlation coefficient, are not met.

Kremer's Rho measures the degree of concordance between two sets of observations. Concordance refers to the extent to which the ranks of the values in one set correspond to the ranks of the values in the other set. If the two sets of observations are perfectly correlated, the Kremer's Rho will be 1. If there is no correlation between the sets, the Kremer's Rho will be 0. A negative Kremer's Rho indicates a negative correlation between the sets.

Compared to other techniques, Kremer's Rho is less sensitive to outliers and can handle non-normal distributions, making it a useful tool in situations where traditional methods may fail.

<h2 style="text-align: center;">Helper Functions</h2>

```python
import pandas as pd
import numpy as np
from typing import List, Tuple, Callable

def run_correlation(df: pd.DataFrame, cols: List[str], corr_func: Callable) -> pd.DataFrame:
    """
    Calculates the pairwise correlations between columns of a pandas DataFrame.
    
    Args:
    df: pandas.DataFrame, the data to be analyzed.
    cols: List[str], a list of column names in the DataFrame to be analyzed.
    corr_func: Callable, a function to calculate the desired correlation between two columns.
    
    Returns:
    pandas.DataFrame, a DataFrame with the correlation values.
    """
    resultant = pd.DataFrame(data=[(1 for i in range(len(cols))) for i in range(len(cols))], 
                             columns=list(cols))
    resultant.set_index(pd.Index(list(cols)), inplace = True)

    for i in cols:
        for j in cols:
            if i != j:
                resultant.loc[i,j] = np.round(corr_func(df[i], df[j]), 2)

def get_date(n: int, ratio: int) -> Tuple[List[str], List[str]]:
    """
    Returns two lists of strings with the length of `n` and `n // ratio`
    
    Args:
    n: int, the length of the first list.
    ratio: int, the ratio used to calculate the length of the second list.
    
    Returns:
    Tuple[List[str], List[str]], a tuple containing two lists of strings.
    """
    x = ['1'] * n + ['2'] * (n // ratio)
    y = ['3'] * n + ['4'] * (n // ratio)
    return x, y
```

<h2 style="text-align: center;" id="correlation7">Cramers V</h2>

Cramer's V is a measure of association between two nominal variables. It is a modification of the chi-square statistic used for contingency tables, which is a table that displays the frequency distribution of two variables. Cramer's V ranges from 0 to 1, where 0 indicates no association between the two variables and 1 indicates a strong association.

Let a sample of size n of the simultaneously distributed variables $$A$$ and $$B$$.

$$n_{i_j}$$ = number of times the values $$(A_{i},B_{j})$$ were observed.
The chi-squared statistic then is:

$${\displaystyle \chi ^{2}=\sum _{i,j}{\frac {(n_{i_j}-{\frac {n_{i}n_{j}}{n}})^{2}}{\frac {n_{i}n_{j}}{n}}}\;}$$

where $${\displaystyle n_{i.}=\sum _{j}n_{ij}}$$ is the number of times the value $$A_{i}$$ is observed and $${\displaystyle n_{.j}=\sum _{i}n_{ij}}$$ is the number of times the value $$B_j$$ is observed.

Cramér's V is computed by taking the square root of the chi-squared statistic divided by the sample size and the minimum dimension minus 1:

$${\displaystyle V={\sqrt {\frac {\varphi ^{2}}{\min(k-1,r-1)}}}={\sqrt {\frac {\chi ^{2}/n}{\min(k-1,r-1)}}}\;,}$$

where:

- $$\varphi$$  is the phi coefficient.
- $$\chi ^{2}$$ is derived from Pearson's chi-squared test
- $$n$$ is the grand total of observations and
- $$k$$ being the number of columns.
- $$r$$ being the number of rows.

[Source](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)

```python
import scipy.stats as ss

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328

    Args:
        x (pd.Series): variable1
        y (pd.Series): variable2

    Returns:
        float: the correletion between variable1 to variable2
    """     
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

n = 500
ratio = 4
x, y = get_date(n, ratio)
print(run_correlation(pd.DataFrame({"x": x, "y": y}), ["x", "y"], cramers_v))
0.99
# skewed dataset
n = 500
ratio = 500
x, y = get_date(n, ratio)
print(run_correlation(pd.DataFrame({"x": x, "y": y}), ["x", "y"], cramers_v))
0.5
# very small dataset
n = 4
ratio = 4
x, y = get_date(n, ratio)
print(run_correlation(pd.DataFrame({"x": x, "y": y}), ["x", "y"], cramers_v))
0
```

<h2 style="text-align: center;" id="correlation8">Theil's U statistic (Uncertainty coefficient)</h2>

Suppose we have samples of two discrete random variables, $$X$$ and $$Y$$. 
By constructing the joint distribution, $$P_{X,Y}(x, y)$$, from which we can calculate the conditional distributions, $$P_{X|Y}(x|y) = P_{X,Y}(x, y)/P_{Y}(y)$$ and $$P_{Y|X}(y|x) = P_{X,Y}(x, y)/P_{X}(x)$$, and calculating the various entropies, we can determine the degree of association between the two variables.
The entropy of a single distribution is given as:

$$H(X)=-\sum _{x}P_{X}(x)\log P_{X}(x)$$

while the conditional entropy is given as:

$$H(X|Y)=-\sum _{x,y}P_{X,Y}(x,y)\log P_{X|Y}(x|y)$$

The uncertainty coefficient or proficiency is defined as:

$$U(X|Y)={\frac  {H(X)-H(X|Y)}{H(X)}}={\frac  {I(X;Y)}{H(X)}}$$

and tells us: given Y, what fraction of the bits of X can we predict? In this case we can think of X as containing the total information, and of Y as allowing one to predict part of such information.

[Source](https://en.wikipedia.org/wiki/Uncertainty_coefficient#:~:text=In%20statistics%2C%20the%20uncertainty%20coefficient,the%20concept%20of%20information%20entropy.)

```python
from typing import Any, List
from collections import Counter

def remove_incomplete_samples(x, y):
    x = [v if v is not None else np.nan for v in x]
    y = [v if v is not None else np.nan for v in y]
    arr = np.array([x, y]).transpose()
    arr = arr[~np.isnan(arr).any(axis=1)].transpose()
    if isinstance(x, list):
        return arr[0].tolist(), arr[1].tolist()
    else:
        return arr[0], arr[1]

def conditional_entropy(x,
                        y,
                        log_base: float = math.e):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    log_base: float, default = e
        specifying base for calculating entropy. Default is base e.
    Returns:
    --------
    float
    """
    x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy

def theils_u(x: List[Any], y: List[Any]) -> float:
    """
    Calculates Theil's U, a measure of inequality of a distribution.
    
    Parameters:
        x (List[Any]): The first list of data.
        y (List[Any]): The second list of data.

    Returns:
        float: The Theil's U value.
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

a = ["a", "b", "c", "d"]
b = ["one", "one", "two", "two"]

example = pd.DataFrame({"A": a, "B": b})
example_cols = []
for col in example.columns:
    
    example[f'{col}_code'] = example[col].astype('category').cat.codes
    example[f'{col}_code'] = example[f'{col}_code'].replace({-1: max(example[f'{col}_code'].unique()) + 1})
    example_cols.append(f'{col}_code')
print(run_correlation(example, example_cols, theils_u))
        A_code  B_code
A_code       1     0.5
B_code       1     1.0
```

<h2 style="text-align: center;">Conclusions </h2>
