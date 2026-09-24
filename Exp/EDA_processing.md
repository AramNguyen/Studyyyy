# **I/ Check the Data & EDA**

## **1/ Define Objectives & Review**

* Define your primary objective (e.g., Prediction or a specific Machine Learning task).
* Review the dataset manually rather than relying entirely on AI agents to maintain full control over feature selection.
* Calculate and determine the size of the sample.

## **2/ Explore Data & Detect Bias**

* Must check out the database by using visualizations to know more about the dataset (Histogram + KDE).
* Calculate the data distribution → by that, detect outliers.
* Check out if the data was biased or not. Deal with data bias by transforming the distributed data into more normalized data (log, log1p, boxcox).

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# overview
df = pd.read_csv('data.csv')
df.info()

# detect outliers and bias using Histogram + KDE
sns.histplot(df['feature'], kde=True)
plt.show()

# transformative the distributed data into more normalized data
df['normalized_feature'] = np.log1p(df['feature'])

```

---

# **II/ Processing Data**

## **1/ Handling Missing Data & Outliers**

* Dealing with the outlier: Mask the data or Impute the data → replace with substitutes values.
* Filling the gaps, identify the type of the blanks.
* Numbers → can be filled by the mean() or med().

```
# missing values
df.isnull().sum()

# replace numerical outliers or blanks with median
median_val = df['numeric_col'].median()
df['numeric_col'] = df['numeric_col'].fillna(median_val)

```

---

## **2/ Handling Categorical Data**

* Objects or String → locate the column we want to clean, identify the method, most common is fill those columns with “Unknown” or “Missing”.


* Step 1: Statistic to get the most - second - third common attribute.
* Step 2: Overwrite the rest of attributes by “Others” and fill the missing gap by “Unknown”. By that way, we aren't afraid of the high dimensionality.

```
# The syntax:
df['colum'] = df['column'].fillna('Unknown')[cite: 4]

# encode categorical data
df = pd.get_dummies(df, columns=['category_col'])

```

---

## **3/ Feature Scaling**

* The feature method: encoding + scaling.
* Data scaling is crucial. Related features must be scaled together so they share the same magnitude.

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

```

---

# **III/ Data Visualization (3 Stages)**

## **1/ Pre-training Stage (Data Exploration & Hypothesis)**

* Visualize the difference between the raw data and the refined data after cleaning.
* It is best to present key statistical metrics such as Standard Deviation, Mean, Median, Variance, etc.
* Formulate a hypothesis about the expected outcomes of the model, and visualize these predictions (e.g., Decision Boundary Plot for Classification or Partial Dependence Plot for Regression).

```
# Key statistical metrics
print(df.describe())
print("Variance:", np.var(df['feature']))

```

---

## **2/ Post-training Stage (Model Evaluation)**

* Create a visualization to illustrate the relationship between your initial hypothesis and the model's actual performance to assess overfitting or underfitting.
* I strongly encourage using a Predicted vs. Actual Plot combined with the Identity Line (y = x).

```
import matplotlib.pyplot as plt
import seaborn as sns

# Predicted vs. Actual Plot with Identity Line
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual (Hypothesis Checking)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Residual plot
sns.residplot(x=y_pred, y=y_test - y_pred)
plt.show()

```

---

## **3/ Post-tuning Stage (Final Assessment)**

* Once the model is tuned, you need a final visualization to demonstrate the improvements gained from your fine-tuning efforts.
* Naturally, the types of graphs used in this stage will be similar to those in Stage 2 to provide a clear before-and-after comparison.

---

# **IV/ Formulate Hypotheses & Advanced Concepts**

## **1/ Final Evaluation & Hypothesis Dashboard**

* Feature Importance: Analyze which individual variable has the greatest impact on your prediction, which has the least impact, and which seems to have absolutely no effect.
* Worst-Case Scenarios: Draw hypothetical conclusions for worst-case scenarios (e.g., extremely overpriced house).
* Performance Thresholds: Establish a Minimum Acceptable Accuracy for your model (e.g., using MAE - Mean Absolute Error) and aim for an R-squared score of over 0.8.

```
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
    
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

```

---

## **2/ Datatypes Categorization**

* Qualitative Data (Categorical):
* Nominal Data: No order (e.g., Gender, Employment_Status).
* Ordinal Data: Has clear ranking (e.g., Burnout_Risk, Sleep_Quality).


* Quantitative Data (Numerical):
* Discrete Data: Countable whole numbers (e.g., Coffee_Cups_Per_Day).
* Continuous Data: Measurements with decimals (e.g., Age, Screen_Time_Hours).



---

## **3/ Distribution Selection (Maximum Likelihood Estimation)**

* Domain Knowledge: Check data boundaries (e.g., cannot be negative -> drop Normal, use Log-Normal/Gamma; counting data -> use Poisson).
* Exploratory Data Analysis (EDA):
* Symmetrical peak -> Normal.
* Right-skewed tail -> Log-Normal or Gamma.
* Steep drop from zero -> Exponential.


* Statistical Testing: Use Q-Q Plot or AIC/BIC scores to evaluate the fit.

```
import scipy.stats as stats

# Q-Q Plot to check distribution fit
stats.probplot(df['feature'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()


```
## **5/ Common Ways to Fine-Tune ML Models (Regularization)**
- In Machine Learning (especially for models like Ridge or Lasso Regression), `alpha` acts as a **Regularization parameter**. Its main job is to control the model's complexity by applying a "penalty" to prevent it from overcomplicating things.

- **When `alpha` is close to 0 (Very Low):** 
  - The penalty is almost non-existent. The model is completely "unleashed" and will bend itself to perfectly connect every single dot in your training data, including random noise. 
  - **The result:** Your training accuracy will look amazingly high, but the model will fail miserably on unseen test data. This is classic **Overfitting** (the model is just memorizing the data instead of understanding the true patterns).

- **When `alpha` increases (High):**
  - The penalty becomes strict. The model's coefficients are squeezed and forced to drop unnecessary complexity, keeping only the core trends.
  - **The result:** Increasing `alpha` is the ultimate medicine to cure Overfitting. However, if `alpha` is pushed *too high*, the model becomes overly rigid and fails to learn anything useful at all. This is called **Underfitting** (the model becomes too restricted to make good predictions).
  
- **The Bottom Line:** A low `alpha` makes the model try too hard to fit the data (causing Overfitting), while a higher `alpha` is used to restrict the model and prevent this memorization. Your ultimate goal is to find the perfect balance.

- Naturally, to evaluate whether the model truly performs well using this hyperparameter tuning method, I highly recommend using a Validation Curve for the most intuitive visualization and straightforward conclusions.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

# 1. Define the range of alpha values to test (from very low to very high)
param_range = np.logspace(-3, 3, 7) # e.g., [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# 2. Calculate accuracy on training and test sets using Cross-Validation
train_scores, test_scores = validation_curve(
    Ridge(), X_train, y_train, param_name="alpha", param_range=param_range, 
    cv=5, scoring="r2"
)

# 3. Calculate mean scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# 4. Plot the Validation Curve
plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label="Training Score (Overfits at low alpha)", color="blue", marker='o')
plt.plot(param_range, test_mean, label="Cross-Validation Score", color="red", marker='s')

plt.title("Validation Curve for Ridge Regression")
plt.xlabel("Alpha (Regularization Parameter)")
plt.ylabel("R^2 Score")
plt.xscale("log") # Use log scale to properly space out alpha values
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
---