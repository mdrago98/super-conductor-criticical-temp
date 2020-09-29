Introduction
============

Loading, Splitting and Cleaning
===============================

#### 

The dataset provided is a compilation of 21,263 super conductor
containing 81 attributes compiled from Japan's National Institute for
Materials Science.

#### Splitting the dataset

The dataset was immediately split into training and testing data, the
latter being discarded away for evaluation purposes. The training
dataset was used during training and development of the models. This was
implemented in the generate\_data.py file which returns these two split
datasets. [\[section:pipeline\]]{#section:pipeline
label="section:pipeline"}

#### Pipeline

A Pipeline by SKLearn was used to automate the cleaning tasks as shown
in Listing [\[lst:pipeline\]](#lst:pipeline){reference-type="ref"
reference="lst:pipeline"}. This would allow the training data to be
cleaned up without inspecting it hence reducing the risk of Data
Leakage. The Dataframe selector is a helper function designed to select
only a subset of the features as to not apply this pipeline to the
output feature.

[\[lst:pipeline\]]{#lst:pipeline label="lst:pipeline"}

``` {.python linenos="" breaklines=""}
def get_pipeline(inputs) -> Pipeline:
    return Pipeline([
        ('selector', DataFrameSelector(inputs)),
        ('std_scaler', StandardScaler()),
    ])


def apply_pipeline(data, x_col, y_col):
    y = data[y_col]
    data = DataFrame(get_pipeline(x_col).fit_transform(data), columns=x_col)
    data['critical_temp'] = y
    x, y = data[x_col], data[y_col]
    return data, x, y
```

Visualising The Data
====================

The describe.py script plots a number of charts attempting to display
the and describe the data. Due to the high number of dimensions a
scatter plot with some of the most important features found later during
development is shown
(Figure [\[fig:scatter\]](#fig:scatter){reference-type="ref"
reference="fig:scatter"}).

[\[fig:scatter\]]{#fig:scatter label="fig:scatter"} ![A scatter matrix
showing the correlations between some of the important
features.](images/scattermatrix.png "fig:")

Figure [\[fig:heat\]](#fig:heat){reference-type="ref"
reference="fig:heat"} gives a general brief overview of all the
variables and their correlation. The map shows that the variables are
highly correlated with one another as shown by the high concentrations
of yellow in the matrix. As the data is so highly correlated,
multicollinearity could be an issue. Multicollinearity is a statistical
phenomenon occurring in multiple regression where a feature could be
predicted linearly from the other features. This could lead to a
situation where the coefficients of the model changing erratically with
small changes in the data. Formally a set of variables are perfectly
collinear when: $$0 = \lambda_1X_{1i} +\dots +\lambda_kX_{ki}$$ for a
regression model
$$Y_1 = \beta_0+\beta_1X_{1i} + \dots + \beta_kX_{ki} + \epsilon$$

[\[fig:heat\]]{#fig:heat label="fig:heat"} ![A heatmap of correlations.
Brighter colors indicate higher correlation.](images/heatmap.png "fig:")

Feature Reduction
=================

Variance Inflation Factor
-------------------------

The Variance Inflation Factor(VIF) is a measure of colinearity of the
input features in multiple regression. It is defined as:
$$VIF_i = \frac{1}{1-R_i^2}$$ This measure was integrated into the
pipeline and the $VIF$ for each of the columns is calculated. As a
general rule columns having a score larger than 10 are
removed [@kutner2005applied].

Principle Component Analysis
----------------------------

Principle component analysis is a statistic procedure that orthogonally
transforms a dataset into uncorrelated variables called principle
components [@pearson1901liii]. The first principle component $x_1$
covers the maximum possible variance in the data such that:
$$w_1 = \underset{||w||=1}{\operatorname{arg}\,\operatorname{max}}\; \left(\sum_i t_i^2 \right) = \underset{||w||=1}{\operatorname{arg}\,\operatorname{max}}\;\left(\sum_i (x_i \cdot w)^2 \right)$$
where $w_k = w_1\dots 1_k$ representing the vectors of the coefficients
and $t_i$ is the vector of principle component scores. The other
components could be calculated subtracting the first $k-1$ principle
components and finding the weight vectors that maximises the variance as
before.

Selecting a Feature Reduction Technique
---------------------------------------

An experiment was devised to score these techniques on the training
dataset to choose the best technique for our testing. The experiment
written in feature\_selection.py, spins up a cross validator, tries each
of these techniques and tables up the results for comparison shown in
Table [\[tab:featurereduction\]](#tab:featurereduction){reference-type="ref"
reference="tab:featurereduction"}. The dataset was standardised using
the pipeline discussed in
Section [\[section:pipeline\]](#section:pipeline){reference-type="ref"
reference="section:pipeline"}. This is important as PCA extracts the
components that maximise the variance, hence having features of
different scale might disproportionately affect the variance messing up
the grouping. The results show that non of the techniques out performed
the benchmark vanilla linear regression. None of the techniques managed
to reduce the number of features to a manageable level indicating that
most of the features are required to capture most of the variance in the
dataset. As a result these techniques were abandoned as they performed
worse than the vanilla linear regression.


  Technique         Cross Validation Score   Number of features
  ----------------- ------------------------ --------------------
  Benchmark         0.7342                   80
  VIF Elimination   0.6523                   29
  PCA               0.5951                   16

  : Scores for the feature reduction techniques.
  []{label="tab:featurereduction"}

Regression Models
=================

In the following section a brief explanation of the models used will be
given along with regularization techniques employed.

Linear Regression
-----------------

Linear regression is a simple linear approach to modelling the
relationship between a dependent variable and it's predictors. Formally
it is described as: $$\begin{aligned}
    y_i &= \beta_0 + \beta_1x_{i1} + \dots + \beta_1x_{ip} + \epsilon_i \\
    y &= X\beta + \epsilon_i && \text{Matrix notation}\end{aligned}$$
where $y$ is a vector of outputs, $X$ is a matrix of row vectors $x_i$
and $x_{i0} = 1~for~i=1,\dots,n$, $\beta$ is a vector of coefficients
and $\epsilon$ is a vector of error terms [@hastie2009elements]. Linear
regression aims to optimise $\beta$ and $\epsilon$ in such a way that
the cost function would be minimised. The cost function could be
described as:
$$\sum_{i=1}^M (y_i-\hat{y_i})^2 = \sum^M_{i=1}\left(y_i-\sum^p_{j=0}w_j \cdot x_{ij}\right)^2$$
where $M$ and $p$

#### Regularisation

Regularisation allows the model to generalise well over unseen samples
in the real world. One problem of the standard linear regression is that
the resulting model would be too complex. In this case linear regression
would have 80 coefficients which could lead to an over-fitted model.

#### Ridge Regularisation

Here the cost function described for Linear Regression is modified in
such a way that a penalty is added to coefficients of large magnitude.
$$\sum_{i=1}^M (y_i-\hat{y_i})^2 = \sum^M_{i=1}\left(y_i-\sum^p_{j=0}w_j \cdot x_{ij}\right)^2 + \lambda\sum^p_{j=0}w^2_j$$
$\lambda$ refers to the penalty term. Setting this to 0 would turn this
cost function into the one used by linear regression minimising this
property.

#### Lasso Regularisation

Like Ridge Regularisation, Lasso builds up on the Linear Regression's
cost function as follows:
$$\sum_{i=1}^M (y_i-\hat{y_i})^2 = \sum^M_{i=1}\left(y_i-\sum^p_{j=0}w_j \cdot x_{ij}\right)^2 + \lambda\sum^p_{j=0}|w_j|$$
It takes account the magnitudes rather than the coefficients hence
leading to zero coefficients and a simpler model.

Random Forest Regression
------------------------

Random Forests is a machine learning technique built on Decision Trees.
Decision Trees are sensitive on the trained data hence if the training
data changes slightly the resulting model could be drastically
different. They also tend to stick to local optima (no back tracing
after splitting the tree) and risk over-fitting. Random Forests employ a
bagging technique (random sampling with replacement) by deploying
multiple decision trees in parallel and returns the mean prediction of
the individual trees. Each tree samples randomly a sample on every split
reducing the chances of over-fitting.

Metrics
-------

#### Root Mean Square Error(RMSE)

The RMSE error for prediction $\hat{y}_t$ and $T$ samples is defined as:
$$\begin{aligned}
    RMSE = \sqrt{\frac{\sum^T_{t=1}(\hat{y}_t-y_t)^2}{T}}\end{aligned}$$
RMSE is frequently used as the error is on the same scale of the output
(in our case Kelvin). Errors are squared before taking the root hence
high errors are penalised.

#### $R^2$

the $R^2$ error or coefficient of determination compares the resultant
model with a constant baseline. The error can range from $-\infty to 1$
where negative values indicate that the model performs worse than the
baseline. $$\begin{aligned}
    R^2 = 1- \frac{MSE(model)}{MSE(baseline)}\end{aligned}$$

Results and Discussion
======================

The evaluation script is located in the evaluate\_models.py file. The
script trains a Linear Regression, Ridge regression and Random forest
using a K-Fold cross validator on the training dataset and then scores
these models using the RMSE and R2 metrics. The script also produces
learning graphs(Figure) and residual plots of the evaluated model and
saves them in the fig directory. The Lasso regression was not included
as this model could not converge. This could be due to a small tolerance
value for the variance of the dataset. This phenomenon was not
investigated further due to time constraints.
Table [\[tab:scores\]](#tab:scores){reference-type="ref"
reference="tab:scores"} shows the scores achieved by the models whilst
Figure [\[fig:residual\]](#fig:residual){reference-type="ref"
reference="fig:residual"} shows the residual plots for each of the
model.

\centering
  Model               Mean Square Error   R2
  ------------------- ------------------- ------
  Linear regression   17.38               0.73
  Ridge Regression    17.9                0.72
  Random Forest       11.96               0.88

  : Scores for the trained models[]{label="tab:scores"}

[\[fig:residual\]]{#fig:residual label="fig:residual"} ![A matrix of
residual plots of the different models.](images/residual.png "fig:")

These results coincide with the results obtained in the original paper
where the RMSE obtained was 17.6 K. and a $R^2$ of
0.74 [@hamidieh2018data]. The Ridge model is also close to the score
obtained by the original paper. The XGBoost Model described in the paper
also out-performs the Random Forest Regressor as it obtains a RSME of
9.5 K.

[\[fig:learningcurve\]]{#fig:learningcurve label="fig:learningcurve"}
![A plot showing the cross validation scores along with training scores
for the random forest.](images/randomforestlearningcurve.png "fig:")

Running Instructions
====================

The project is structured as follows:

-   utils/ - containing utilities such as plotting helper functions;

-   figs/ - the directory where the figures get saved;

-   data/ - a directory containing the datasets (train, test and
    original);

-   describe.py - a script that performs descriptive analytics on the
    dataset;

-   evaluate\_models.py - evaluates the trained models;

-   feature\_selection.py - A script that performs feature selection
    analysis

Listing [\[lst:running\]](#lst:running){reference-type="ref"
reference="lst:running"} shows how to run the scripts.

[\[lst:running\]]{#lst:running label="lst:running"}

``` {.sh linenos="" breaklines=""}
 venv/bin/python <file_name>
```

#### Dependencies

-   Scikit Learn [@scikitlearn]

-   Numpy [@2020SciPyNMeth]

-   Matplotlib [@hunter2007matplotlib]

-   Statsmodels (Used for calculating the variance inflation
    factor) [@seabold2010statsmodels]
