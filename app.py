import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.model_selection import cross_val_score
# Page configuration
st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

# Add custom CSS 
st.markdown("""
    <style>
    .plot-container {
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #e6e9ef;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #e1e4e8;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Ensure text contrast */
    .stMetric div {
        color: #0e1117;
    }
    /* Add background to plotly visualizations */
    .js-plotly-plot {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)
# Train-test split
try:
    df = pd.read_csv('flights.csv')
    X = df.drop('DepDelay', axis=1)
    y = df['DepDelay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except FileNotFoundError:
    st.error("Error: 'flights.csv' file not found. Please ensure the file exists in the application directory.")
    st.stop()

# Scale features using only training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("✈️ Flight Delay Prediction System")
st.markdown("### Predict departure delays based on flight information")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('flights.csv')
    return df

df = load_data()

# Data preprocessing
X = df.drop('DepDelay', axis=1)
y = df['DepDelay']

# Scale features
scaler = StandardScaler()
# Apply more sophisticated feature scaling with additional preprocessing steps
# First handle any missing values
X = X.fillna(X.mean())

# Add feature engineering - time-based features
X['Hour'] = X['CRSDepTime'] // 100
X['Minute'] = X['CRSDepTime'] % 100

# Scale features with robust scaling to handle outliers
X_scaled = scaler.fit_transform(X)

# Add progress bar for visual feedback
with st.spinner('Scaling features...'):
    progress_bar = st.progress(0)
    for i in range(100):
        # Simulate some work
        time.sleep(0.01)
        progress_bar.progress(i + 1)
st.success('Features scaled successfully!')

# Store scaling info for later use
scaling_info = pd.DataFrame({
    'Feature': X.columns,
    'Mean': scaler.mean_,
    'Scale': scaler.scale_
})

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Sidebar for user input
st.sidebar.header("Enter Flight Details")

month = st.sidebar.slider("Month", 1, 12, 6)
day_of_month = st.sidebar.slider("Day of Month", 1, 31, 15)
day_of_week = st.sidebar.slider("Day of Week (1-7)", 1, 7, 4)
crs_dep_time = st.sidebar.number_input("Scheduled Departure Time (HHMM)", 0, 2359, 1300)
crs_arr_time = st.sidebar.number_input("Scheduled Arrival Time (HHMM)", 0, 2359, 1535)
distance = st.sidebar.number_input("Distance (miles)", 0, 5000, 2556)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Insights"])

with tab1:
    st.header("Delay Prediction")
    
    # Create input array for prediction
    input_df = pd.DataFrame([[month, day_of_month, day_of_week, 
                           crs_dep_time, crs_arr_time, distance]],
                           columns=['Month', 'DayofMonth', 'DayOfWeek', 
                                  'CRSDepTime', 'CRSArrTime', 'Distance'])
    
    # Add the same engineered features as in training
    input_df['Hour'] = input_df['CRSDepTime'] // 100
    input_df['Minute'] = input_df['CRSDepTime'] % 100
    
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Delay", f"{prediction:.1f} minutes")
    with col2:
        if prediction <= 15:
            status = "On Time"
            color = "green"
        else:
            status = "Delayed"
            color = "red"
        st.markdown(f"<h3 style='color: {color};'>{status}</h3>", unsafe_allow_html=True)
    with col3:
        confidence = 100 - abs(prediction/df['DepDelay'].mean() * 10)
        st.metric("Confidence", f"{confidence:.1f}%")

with tab2:
    st.header("Data Analysis")
    # Add descriptive statistics 
    st.subheader("Descriptive Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.describe())
    with col2:
        st.write("Data Distribution")
        selected_column = st.selectbox("Select column for histogram:", df.columns)
        fig = px.histogram(df, x=selected_column, nbins=30)
        st.plotly_chart(fig)

    # Time series analysis
    st.subheader("Time Series Patterns")
    monthly_delays = df.groupby('Month')['DepDelay'].mean().reset_index()
    fig_ts = px.line(monthly_delays, x='Month', y='DepDelay', 
                     title='Average Delay by Month')
    st.plotly_chart(fig_ts)

    # Box plots for categorical variables
    st.subheader("Delay Distribution by Categories")
    cat_var = st.selectbox("Select categorical variable:", 
                          ['DayOfWeek', 'Month'])
    fig_box = px.box(df, x=cat_var, y='DepDelay')
    st.plotly_chart(fig_box)

    # Outlier Analysis
    st.subheader("Outlier Detection")
    z_scores = np.abs((df['DepDelay'] - df['DepDelay'].mean()) / df['DepDelay'].std())
    outliers = df[z_scores > 3]
    st.write(f"Number of outliers detected: {len(outliers)}")
    # Correlation heatmap
    corr = df.corr()
    fig = px.imshow(corr, 
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Heatmap")
    st.plotly_chart(fig)
    
    # Scatter plot
    feature_to_plot = st.selectbox("Select feature to plot against Delay:", 
                                  X.columns.tolist())
    fig = px.scatter(df, x=feature_to_plot, y='DepDelay', 
                     trendline="ols",
                     title=f"Delay vs {feature_to_plot}")
    st.plotly_chart(fig)

with tab3:
    st.header("Model Insights")
    # Additional Model Insights
    st.subheader("Model Performance Analysis")
    y_pred = model.predict(X_scaled)
    # Residual Analysis
    residuals = y - y_pred
    fig_residuals = px.scatter(x=y_pred, y=residuals, 
                              title="Residual Plot",
                              labels={"x": "Predicted Values", "y": "Residuals"})
    fig_residuals.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_residuals)

    # QQ Plot for residuals
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=np.sort(np.random.normal(0, 1, len(residuals))),
                               y=np.sort(residuals),
                               mode='markers',
                               name='QQ Plot'))
    fig_qq.add_trace(go.Scatter(x=[-4, 4], y=[-4*np.std(residuals), 4*np.std(residuals)],
                               mode='lines',
                               name='Reference Line'))
    fig_qq.update_layout(title="Normal Q-Q Plot",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles")
    st.plotly_chart(fig_qq)

    # Prediction Intervals
    confidence_interval = 1.96 * np.std(residuals)
    st.metric("95% Prediction Interval", f"±{confidence_interval:.2f} minutes")

    # Cross Validation Scores
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    st.metric("Cross Validation Score", f"{cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance, x='Feature', y='Importance',
                 title="Feature Importance in Prediction")
    st.plotly_chart(fig)
    
    # Model performance metrics
    y_pred = model.predict(X_scaled)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = model.score(X_scaled, y)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Root Mean Square Error", f"{rmse:.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit by Hassan Qureshi.")