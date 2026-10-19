# What Drives the Price of a Used Car?

**Notebook:** [Practical_Application_2.ipynb](Practical_Application_2.ipynb)

## Business Problem

A used car dealership wants to know what consumers value in a used car so it can price and stock its inventory more effectively. This project analyzes about 427,000 used-car listings (a Kaggle/Craigslist dataset) to find which vehicle features are most strongly associated with asking price.

## Data Science Task

This is a supervised regression problem: predict "price" from vehicle attributes such as model year, odometer reading, cylinders, fuel type, drive type, manufacturer, condition, and state.

## Approach (CRISP-DM)

1. **Data understanding:** Checked missing values, duplicates, invalid values, and distributions.
2. **Data preparation:**
   - Removed repeat listings (the same VIN or the same vehicle details reposted under new IDs).
   - Dropped unneeded columns ("id", "VIN", "size", "region", "model").
   - Filtered unrealistic values: price outside $500 to $100,000, year before 1990, odometer outside 1 to 500,000 miles.
   - Cleaned "cylinders" and filled missing categories with "unknown."
   - The cleaned dataset has 205,718 rows and 13 columns.
3. **Modeling:** Compared Linear Regression, Ridge, Lasso, and a Decision Tree using pipelines, 3-fold cross-validated grid search, and a 75/25 train/test split.
4. **Evaluation:** Compared models on test R², MAE, and RMSE.

## Model Results

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.655 | $5,167 | $7,757 |
| Ridge | 0.655 | $5,167 | $7,756 |
| Lasso | 0.655 | $5,167 | $7,757 |
| **Decision Tree** | **0.741** | **$4,052** | **$6,722** |

The Decision Tree was the best model. On average its price predictions were off by about $4,000.

## Key Findings

- **Model year is the strongest driver of price.** It accounts for about 52% of the Decision Tree's importance. Newer cars sell for more.
- **Engine size (cylinders) and mileage come next**, at about 16% and 10%. Bigger engines and lower mileage go with higher prices.
- **Diesel vehicles carry a clear premium**, with about 8% importance in the tree and one of the largest effects in the linear model.
- **Drive type matters.** 4WD and front-wheel-drive vehicles show up as meaningful predictors.
- **Brand and body type matter.** Luxury and electric brands (for example Tesla and Porsche) and pickups sell at premiums, all else equal.

## Recommendations for the Dealership

- Build inventory around newer, lower-mileage vehicles. These command the highest prices.
- Pay attention to diesel trucks and 4WD vehicles, which carry premiums.
- Use the model for a starting price estimate, then adjust for local demand, vehicle condition, and competing listings.

## Limitations and Next Steps

- Some limitations are that prices are asking prices from listings and not the final sale prices. Also, the dataset is Craigslist data so there were many duplicate listings and inadequately categorized listings.
- Over half the original rows were removed during cleaning, so findings apply to listings with reasonably complete information.
- Next steps: Model using Random Forest or gradient boosting, bring the `model` column back as a feature, and validate against actual sale prices.
