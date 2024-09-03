<h1>Linear Regression</h1>

<h2>Languages and Utilities Used</h2>

- <b>Pandas</b>
- <b>Matplotlib</b>
- <b>seaborn</b>
- <b>numpy</b>
- <b>math</b>
- <b>statsmodels</b>

<h2>Data Source</h2>

The data set is appended to this repository and contains six sheets of data for working with linear regression.

<h2>Introduction and concepts</h2>

<h3>Simple linear regression</h3>

Linear regression analysis is a powerful method for modeling a linear relationship between two or more variables. The objective is to **estimate the value of a random variable** given that we know the value of an associated variable. A linear regression follows the equation:

$$y = \beta_0 + \beta_1 x$$

where: 
  
  - $x$ is the **predictor**, often called the independent or explanatory variable 
  - $y$ is the **response**, or dependent variable
  - $\beta_0$ is the **intercept**
  - $\beta_1$ is the **slope**, or regression coefficient. 

The set of parameters estimates are referred to as a **regression model**. Sometimes, The linear regression **model** is expressed as follows:

$$\hat{y} = \beta_0 + \beta_1 x$$

This is to highlight that $\hat{y}$ is the estimate of $y$, not its observed value. The fitted value is just the estimate of $y$ and generally differs from the observed value. This fact is often expressed as follows:

$$\hat{y}_i = y_i + e_i\,$$

where $i$ denotes the i-th observation of ($x_i, y_i$) and $e_i$ is called the **residual**. Residuals express difference between the predicted values $\hat{y}$ and the observed values $y$. In general, the better the regression model, the closer to zero the residuals are.

<h3>Visual interpretation</h3>

<p align="center">
Regression line example<br/>
<img src="https://i.imgur.com/jzESgBQ.png" height="50%" width="50%" alt="Regression line example"/>
<br />
</p>

In a simple linear regression we are trying to find a line that fits a set of points in a plane defined by $x$ and $y$, so that the total distance between the points and the lines is as small as possible. This line is called the **regression line**.

The intercept $\beta_0$ is then equal to the distance between the intersection of the $x$ and $y$ axes and the intersection between the regression line and the $y$ axis. 
The slope $\beta_1$ is then equal to the tangent of the angle $\alpha$ between the regression line and the $x$ axis: $\beta_1 = \tan\alpha$. When the slope $\beta_1$ equals $1$, $\alpha$ is equal to $45^\circ$.

<h3>Underlying assumptions</h3>

The successful application of a linear regression requires several assumptions regarding $x$ and $y$ to be satisfied. The most important of these are: 
 - **Linearity**, which means that the independent variable $x$ and the dependent variable $y$ are linearly associated 
 - **Constant variance** of the response variable $y$ regardless of the value of the independent variable $x$
 - **Independence of errors**, which means that the errors $e$ are not correlated with each other.
 
For more details about the assumption of linear regression analysis please see: https://en.wikipedia.org/wiki/Linear_regression#Assumptions

<h3>Fitting a linear model</h3>

A quick estimation of the regression parameters can be performed as follows:
 - Slope
$\beta_1 = \frac{\sigma_y}{\sigma_x}\,R_{x,y}$
 - Intercept
$\beta_0 = \bar{y} - \beta_1 \bar{x}$

where:
 - $\sigma_x$, $\sigma_y$ denote the standard deviations of $x$ and $y$ respectively
 - $\bar{x}$ and $\bar{y}$ are means of $x$ and $y$ respectively
 - $R_{x,y}$ is the Pearson's coefficient of correlation between $x$ and $y$
 
In a real life, such estimation of the regression parameters is rarely performed as more precise computational methods can be used instead. However, the above equations provide a good hint about the association between the regression parameters and the basic statistical properties of the $x$ and $y$.

<h2>Ordinary least square method</h2>

To estimate values of $y$ we are looking for values of $\beta_0$ and $\beta_1$ that make residuals as small as possible.

The most common practice is to choose the line that minimizes the sum of the squared residuals; this technique is called **ordinary least squares (OLS)**. 

Mathematically this can be formulated as follows:

$\hat{\beta} = argmin_{\beta}\, (\sum e^2_i$)

where (reminder):
  - $\widehat{\beta}$ denotes the estimate of the regression parameters (slope $\beta_1$ and intercept $\beta_0$)
  - $i$ is the index of the observation (i.e. pair of $x_i, y_i$ values)
  - $e_i$ is the residual associated with the $i$-th observation
  - $argmin_{\beta}$ denotes the procedure of searching for the estimates of $\beta_0$ and $\beta_1$ that minimizes the sum of the squared residuals

An important consequence of this approach is that it gives a higher weight to points with higher residuals (because the residuals are squared). The regression parameters (intercept and slopes) estimated using the OLS methodology are considered **"optimal"**. The resulting regression line is then often referred to as **least-squares line**.


<h2>How to evaluate a model</h2>

*Every regression model must be subjected to examination to evaluate its validity* 

i.e., its ability to provide meaningful estimates of $y$. Examination of the regression model typically involves multiple steps.

<h3>Examination of regression residueals</h3>

The residuals of the valid regression model must be *(nearly) normally distributed and centered close to 0.* Failure to meet this requirement may indicate a mis-specified model, i.e., the given explanatory variable is not sufficient to estimate $y$, or may indicate the presence of influential points (discussed later). If the mean is not zero the model is said to be **biased**. 

If observations are independent, i.e. there is no underlying structure in the data, *the variability of the residuals must be constant across the whole dataset.* 

A quick way to assess the residuals is to inspect them visually. The below figures demonstrate several examples of invalid regression caused by violation of the regression assumptions/criteria (from left to right):

  - Relationship is not linear
  - Presence of influential points
  - Residual variability is not constant
  - Residuals are auto-correlated

<p align="center">
<img src="https://i.imgur.com/CNWWtdw.png" height="80%" width="80%" alt="Evaluation by visual inspection"/>
</p>

A more rigid visual technique to examine the model's residuals is to generate the **normal probability plot**. The normal probability plot compares quantiles of the actual residuals against the theoretical one, i.e., quantiles of the normally distributed variable with zero mean and standard deviation equal to the one of the actual residuals. 

Ideally, the residuals should be normally distributed and all the point in the plot should lie on the identity line (line that forms a 45° angle with the x-axis). Deviations from the identity line suggest departures from normality.

<p align="center">
<img src="https://i.imgur.com/qSxFmjy.png" height="80%" width="80%" alt="Normal probability plot"/>
</p>

<h3>Root mean square error</h3>

To assess how good the estimates of $y$ are, we often use **root mean square error (RMSE)**, which is the square root of the mean of the squares of the model residuals:

$$RMSE = \sqrt{\frac{1}{N}\sum^N_i e^2_i},$$

where $N$ indicates the number of observations. 

To facilitate a comparison between datasets, or models, with different scales, it is useful to **normalize** the values of RMSE by dividing by the mean of $y$ ($\bar{y}$):

$$NRMSE = \frac{RMSE}{\bar{y}},$$

In some cases it is more desirable to normalize the values of RMSE by dividing by the range of the response variable ($y_{max} - y_{min}$) can be more desirable:

$$NRMSE = \frac{RMSE}{y_{max} -\, y_{min}}.$$

<h3>Coefficient of determination</h3>

Another common way to assess the regression model is to calculate the correlation $R$ between the estimates $\widehat{y}$ and the actual observations $y$. Ideally the estimates and actual values of $y$ are perfectly correlated leading to a correlation coefficient equal to $1$ ($R = 1$). In practice this (almost) never happens. 

The correlation coefficient can take a range from $-1$ to $1$, where negative values indicate an inverse association between the estimates and the actual observations, which in practice happens only rarely (the model fit would have to be exceptionally bad). It is therefore more practical to use its square, called **coefficient of determination**, or simply R-squared ($R^2$). 

The $R^2$ provides a measure of how well the observed values are replicated by the regression model, based on the proportion of total variation of outcomes explained by the model. The values of R-squared range from 0 to 1, where $R^2 = 0$ indicates that there is no linear relationship between the model estimates and the observed values, while $R^2 = 1$ indicates that the model provides a perfect fit to the observed values.

Importantly, $R^2$ can be calculated as:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}},$$

where $SS_{res}$ is the sum of squares of the model residuals $e$, often called the residual sum of squares; and $SS_{tot}$ is the total sum of squares: 

$$SS_{res} = \sum^N_i(\widehat{y}_i - y_i)^2$$

$$SS_{tot} = \sum^N_i(y_i - \bar{y})^2$$

In the case of linear regression, the total sum of squares can be easily partitioned into what is called explained variance ($SS_{explained}$) and residual sum of squares ($SS_{res}$):

$$SS_{tot} = \sum^N_i(y_i - \bar{y_i})^2 = \sum^N_i(\widehat{y}_i + e_i - \bar{y})^2 = \sum^N_i(\widehat{y}_i - \bar{y} + e_i)^2 = \sum^N_i(\widehat{y}_i - \bar{y})^2 + \sum^N_i e^2_i = SS_{explained} + SS_{res}$$

From this we obtain:

$$R^2 = 1 - \frac{SS_{res}}{SS_{explained} + SS_{res}}$$

which implies that if:

- $R^2 = 1 \implies SS_{explained} = SS_{total} \implies SS_{res} = 0$, i.e. the model explains all the variability in the observations,
- $R^2 = 0 \implies SS_{res} = SS_{total} \implies SS_{explained} = 0$, i.e. the model explains nothing (no variability)

Proof of the above computation can be found here: https://en.wikipedia.org/wiki/Partition_of_sums_of_squares

<h2>Understanding regression output from software</h2>

Most software provides a standardized summary of the regression analysis, typically containing several measures for the evaluation of the quality of the performed regression. Let's take a look at the output from the previous example:

| parameter| value | parameter | value |
| --- |	--- | --- |	--- |
| **Dep. Variable:** |	y | **R-squared:** |	0.895 |
| **Model:** |	OLS	| **Adj. R-squared:** |	0.880 |
| **Method:** |	Least Squares |	**F-statistic:**	| 59.44 |
| **Date:** |	Wed, 08 Aug 2018 |	**Prob (F-statistic):** |	0.000115 |
| **Time:** |	12:00:31	| **Log-Likelihood:** |	-12.056 |
| **No. Observations:** |	9 |	**AIC:** |	28.11 |
| **Df Residuals:** |	7 |	**BIC:** |	28.51 |
| **Df Model:** |	1 |		 **Covariance Type:** |	nonrobust |

The left side of the summary provides a description of the performed regression:

 - **Dep. variable** shows which data column served as the dependent variable (response)
 - **Model** indicates the type of the regression analysis (in this case ordinary least squares)
 - **Method** is the method used to fit the regression parameters
 - **Date/Time** is data and time when the regression analysis was performed
 - **No. Observations** is the number of observations
 - **Df Residuals**  is the number of residual degrees of freedom, defined as the number of observations minus one, minus the model's degrees of freedom (see below).
 - **Df Model** is the number of degrees of freedom in the model, for simple linear regression this is always 1.
 - **Covariance Type** indicates whether robust regression analysis was applied (for more details visit: https://en.wikipedia.org/wiki/Robust_regression)
 
The right side provides an evaluation of the regression analysis, i.e. how well the model fits the data, where:

 - **R-squared** is the coefficient of determination ($R^2$), which describes the amount of variation in the response that is explained by the model.
 - **Adj. R-squared** indicates the adjusted coefficient of determination. This is necessary because, when you use multiple predictors, (the subject of the next module), any addition will always improve the $R^2$  because the degrees of freedom decrease.  This indicator balances the number of observations vs. the number of predictors.  In practice, be cautious if $R^2$ and adjusted $R^2$ are very different as this may indicate overfitting (will be discussed later).
 - **F-statistic** is the value of the F-statistic (sometimes referred to as the F-value), that is calculated as:
 
$$F = \frac{SS_{explained}}{SS_{res}} \frac{n - k}{k - 1}$$
 
 where $n$ is the total number of observations and $k$ is the number of degrees of freedom in the model.
 
 - **Prob (F-statistic)** is often called the p-value, and indicates the probability that the true values of the slope and intercept are both equal to zero.
 - **Log-Likelihood** is the natural logarithm of the estimated maximum value of the **likelihood function** of the model. Likelihood function (or simply likelihood) is the probability that the true values of $beta$ are equal their estimated values $\hat{\beta}$ given the set of observations. We will discuss the the likelihood function more thoroughly in the next module.
 - **AIC**  and **BIC** are the values of Aikake information criterion (AIC) and Bayesian information criterion (BiC). AIC and BIC are an estimators of the relative quality of the statistical models for a given set of data, they are calculated as:
 
$$\text{AIC} = 2k - 2\ln(\hat{L})$$
 
 and 
 
$$\text{BIC} = 2\ln(n)k - 2\ln(\hat{L})$$
 
 where $n$ is the total number of observations, $k$ is the number of degrees of freedom in the model and $\hat{L}$ is the estimated maximum of the likelihood function of the model. Lower are the values of AIC and BIC the greater is the relative quality of the model.

<h3>Testing regression parameters</h3>

 In most software, the regression parameters are automatically subjected to statistical testing where the null hypothesis is that the true values of these parameters are equal to a certain value which we refer to as the null value (this is a different use of the term than in computer science). Here null means a situation that is considered to be ordinary and typically is interpreted as a failure of the regression model. In most cases we are interested if the estimates are different from zero. In addition, most software provides an evaluation of the uncertainty of the obtained estimates, such as standard deviation and 95% confidence interval.

| parameter| coef | std err | t | P > $t$| 	[0.025 | 0.975] |
| --- | --- | --- | --- | --- | --- | --- |
| **Intercept** | 0.5043 |	0.720 |	0.701 |	0.506 |	-1.197 |	2.206
| **x** | 0.9290 |	0.120 |	7.710 |	0.000 |	0.644 |	1.214


 - **coef** is the estimated value of the regression parameters: $\hat{\beta_0}$ and $\hat{\beta_1}$.
 - **std err** is the standard error of the parameter estimate, often denoted as $s.e.(\hat{\beta_0})$ and $s.e.(\hat{\beta_1})$
 - **t** is the value of the **t-statistic**, which is calculated as:
 
$$t_{\hat{\alpha}} = \frac{\hat{\alpha} - \alpha_0}{s.e.(\hat{\alpha})}$$
 
$$t_{\hat{\beta}} = \frac{\hat{\beta} - \beta_0}{s.e.(\hat{\beta})}$$
 
 where $\alpha_0$ and $\beta_0$ are null values against which we are testing the estimates. In this case (and in most cases) they are equal to zero.
 - **P > $|t|$** - is the p-value associated with the given parameter, i.e. the probability that the true value of the given regression parameter is equal to the null value, given the set of observations. If the null value is equal to zero, the p-value of the slope can be interpreted as a probability that the $x$ and $y$ are not linearly associated.
 - The remaining two columns of the output show the **95% confidence interval** of the given regression parameter (for more details visit: https://en.wikipedia.org/wiki/Confidence_interval)

<h2>Outliers in linear regression</h2>

**Outliers** in a regression are observations that fall far from the "cloud" of ($x$, $y$) points.  These points are especially important because they can have a strong influence on the least squares line. Points that fall horizontally away from the center of the cloud tend to pull harder on the line, so we call them points with high **leverage**. If one of these high leverage points appears to actually invoke its influence on the slope of the regression line then we call it an **influential point**. Usually we can say that a point is influential if, had we fitted the line without it, the influential point would have been unusually far from the least squares line.

<h3>Types of outlier</h3>

 - Outlier far from the other points, though it only appears to slightly influence the line (Figure 1).
 - Outlier on the right, though it is quite close to the least-squares line, which suggests that it isn't very influential (Figure 2).
 - Point far away from the cloud, and this outlier appears to pull the least-squares line up on the right; examine how the line around the primary cloud doesn't appear to fit very well.

<p align="center">
<img src="https://i.imgur.com/NuFKlYb.png" height="80%" width="80%" alt="Types of outlier"/>
<br />
</p>

 - There is a primary cloud and then a small secondary cloud of four outliers (Figure 4). The secondary cloud appears to be influencing the line somewhat strongly, making the least-squares line fit poorly almost everywhere.
 - There is no obvious trend in the main cloud of points and the outlier on the right appears to largely control the slope of the line (Figure 5).
 - Outlier far from the cloud, however, it falls quite close to the least-squares line and does not appear to be very influential (Figure 6).

<h3>A word of caution with outliers</h3>

**Don't ignore outliers when fitting a final model.** If there are outliers in the data, they should not be removed or ignored without a good reason. Whatever final model is fit to the data would not be very helpful if it ignores the most exceptional cases. Important information may be contained in the outliers.

The objective is not to get a good fit, it is to understand the process you are modeling and to create an useful model. By eliminating outliers you are missing out the opportunity to understand different scenarios you may encounter in the future.

















































