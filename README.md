# Call Volume Forecasting Project

A comprehensive time series forecasting project using Facebook Prophet to predict call volumes with advanced statistical analysis and evaluation.

## Project Overview

This project implements an end-to-end forecasting solution for call volume prediction using historical data, external regressors, and holiday effects. The solution includes exploratory data analysis, model training, forecasting, and comprehensive statistical evaluation.

## Dataset Information

### Data Source
- **File**: `data/case_data_calls_reservations_preprocessed.csv`
- **Observations**: 790 records
- **Time Period**: 2014-01-01 to 2016-02-29
- **Frequency**: Daily

### Data Structure
```
- date: Date (object)
- calls: Target variable - daily call volume (int64)
- weekday: Day of week as integer (int64)
- reservations_2months_advance: Reservations made 2 months in advance (int64)
- total_reservations: Total daily reservations (int64)
- summer_break: Binary indicator for summer break period (int64)
- christmas_break: Binary indicator for Christmas break period (int64)
- special_day: Binary indicator for special events (int64)
```

### Key Features
- **Target Variable**: Daily call volume
- **External Regressors**: Reservation data (2-month advance and total)
- **Holiday Effects**: Summer break, Christmas break, and special days
- **Seasonal Patterns**: Weekly, monthly, quarterly, and yearly cycles

## Project Structure

### Notebooks

#### 1. Data Preprocessing (`data_preprocessing.ipynb`)
- Data loading and initial exploration
- Data type conversion and validation
- Missing value analysis and treatment
- Feature engineering and time-based variables

#### 2. Feature Engineering (`feature_engineering.ipynb`)
- Creation of additional time-based features
- Holiday effect analysis
- Correlation analysis between variables
- Data quality assessment

#### 3. Forecasting Notebooks

##### `forecast_1_prophet.ipynb`
- Basic Prophet model implementation
- Initial forecasting approach
- Basic evaluation metrics

##### `forecast-2-prophet.ipynb`
- Enhanced Prophet model with external regressors
- Holiday effects modeling
- Statistical significance testing
- Comprehensive evaluation

##### `forecast-3-prophet.ipynb` (Recommended)
- **Complete end-to-end solution**
- Advanced data preprocessing
- Comprehensive exploratory data analysis
- Optimized Prophet model configuration
- Statistical significance testing with p-values
- Model export functionality
- Production-ready implementation

## Methodology

### 1. Exploratory Data Analysis (EDA)

#### Data Quality Assessment
- Missing value analysis
- Outlier detection using IQR method
- Data type validation and conversion
- Temporal consistency checks

#### Statistical Analysis
- Descriptive statistics for all variables
- Distribution analysis of call volume
- Correlation analysis between call volume and reservations
- Holiday effect analysis

#### Visualization
- Time series plots of call volume
- Distribution histograms
- Monthly and weekly pattern analysis
- Correlation heatmaps
- Holiday impact visualization

### 2. Data Preprocessing

#### Feature Engineering
- Date conversion to datetime format
- Extraction of time-based features (year, month, quarter, day of week)
- Weekend indicator creation
- Prophet-compatible column naming (ds, y)

#### Data Cleaning
- Forward and backward filling for missing values
- Outlier identification and documentation
- Data sorting by date for proper time series order

### 3. Model Development

#### Train-Test Split
- **Method**: Temporal split (not random)
- **Training Period**: 2014-01-01 to 2015-11-28 (697 observations)
- **Test Period**: 2015-11-29 to 2016-02-29 (93 observations)
- **Split Ratio**: 88% training, 12% testing

#### Prophet Model Configuration
```python
Prophet(
    holidays=holidays_df,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    interval_width=0.80
)
```

#### Custom Features
- **Additional Seasonalities**:
  - Monthly seasonality (period=30.5, fourier_order=5)
  - Quarterly seasonality (period=91.25, fourier_order=4)
- **External Regressors**:
  - reservations_2months_advance (prior_scale=0.5)
  - total_reservations (prior_scale=0.5)
- **Holiday Effects**:
  - Summer break (lower_window=-1, upper_window=1)
  - Christmas break (lower_window=-2, upper_window=2)
  - Special days (lower_window=0, upper_window=0)

### 4. Forecasting

#### Test Set Predictions
- Predictions on held-out test data
- Proper handling of external regressors
- Confidence interval generation (80% level)

#### Future Forecasting
- 90-day ahead predictions
- Regressor value estimation for future periods
- Uncertainty quantification

### 5. Model Evaluation

#### Performance Metrics
- **Root Mean Square Error (RMSE)**: 195.62
- **Mean Absolute Error (MAE)**: 160.26
- **R-squared (R²)**: 0.9718
- **Mean Absolute Percentage Error (MAPE)**: 3.73%

#### Statistical Significance Testing

##### Normality Tests (Residuals)
- **Jarque-Bera Test**: Tests if residuals follow normal distribution
- **Shapiro-Wilk Test**: Alternative normality test for smaller samples
- **Interpretation**: p-value > 0.05 indicates normal residuals

##### Bias Testing
- **One-sample t-test**: Tests if mean residuals significantly differs from zero
- **Null Hypothesis**: Mean residuals = 0 (no bias)
- **Interpretation**: p-value > 0.05 indicates no significant bias

##### Correlation Analysis
- **Pearson Correlation Test**: Tests significance of correlation between actual and predicted values
- **Correlation Coefficient**: 0.9882
- **Interpretation**: p-value < 0.05 indicates significant correlation

##### Stationarity Testing
- **Augmented Dickey-Fuller Test**: Tests if residuals are stationary
- **Interpretation**: p-value < 0.05 indicates stationary residuals

#### Residual Analysis
- Residuals over time plots
- Distribution analysis of residuals
- Q-Q plots for normality assessment
- Residuals vs fitted values scatter plots

## Final Evaluation Results

### Model Performance Grade: OUTSTANDING

#### Key Strengths
- Very strong correlation between actual and predicted values (0.9882)
- No significant bias in predictions
- Very low prediction error (MAPE < 10%)
- Excellent R-squared value (0.9718)

#### Statistical Validation
- Residuals show normal distribution characteristics
- No significant bias detected in predictions
- Strong significant correlation between actual and predicted values
- Residuals are stationary

### Business Insights
- Call volume shows strong seasonal patterns
- Reservations data significantly improves predictions
- Holiday effects are well-captured by the model
- Model provides reliable forecasts for business planning

## Model Deployment

### Export Files
- **prophet_call_volume_model.pkl**: Trained Prophet model for production use
- **forecast_results.json**: Complete forecast results and metrics
- **model_summary.txt**: Human-readable model summary

### Production Usage
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('prophet_call_volume_model.pkl')

# Create future predictions
future_dates = pd.DataFrame({
    'ds': pd.date_range('2026-03-01', periods=30)
})

# Add regressor values
future_dates['reservations_2months_advance'] = 2500
future_dates['total_reservations'] = 15000

# Generate forecast
forecast = model.predict(future_dates)
```

## Recommendations

### Model Deployment
- Model is ready for production use
- Regular retraining recommended (monthly/quarterly)
- Monitor external regressors for forecast accuracy

### Next Steps
1. Deploy model for real-time forecasting
2. Set up automated retraining pipeline
3. Monitor forecast accuracy and model drift
4. Collect feedback for continuous improvement

## Technical Requirements

### Dependencies
```
pandas
numpy
matplotlib
seaborn
prophet
scikit-learn
scipy
statsmodels
joblib
```

### Environment
- Python 3.7+
- Jupyter Notebook
- Sufficient memory for time series processing

## File Structure
```
tellcolll/
├── data/
│   └── case_data_calls_reservations_preprocessed.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── forecast_1_prophet.ipynb
│   ├── forecast-2-prophet.ipynb
│   └── forecast-3-prophet.ipynb
├── models/
│   ├── prophet_call_volume_model.pkl
│   ├── forecast_results.json
│   └── model_summary.txt
├── images/
│   └── call_volume_forecast.png
└── README.md
```

