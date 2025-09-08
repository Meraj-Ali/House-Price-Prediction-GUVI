import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    #layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('enhanced_house_price_dataset.csv')
    return df

# Main App
def main():
    st.title("🏠 House Price Prediction & Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the App Mode",
        ["📊 Data Analysis & Visualization", "🎯 House Price Prediction"]
    )
    
    if app_mode == "📊 Data Analysis & Visualization":
        data_analysis_section(df)
    elif app_mode == "🎯 House Price Prediction":
        prediction_section(df)

# PART 1: Data Analysis & Visualization
def data_analysis_section(df):
    st.header("📊 Data Analysis & Visualization")
    
    # Dataset Overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns)-1)
    with col3:
        st.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
    with col4:
        st.metric("Price Range", f"₹{df['Price'].max() - df['Price'].min():,.0f}")
    
    # Data Preview
    st.subheader("🔍 Data Preview")
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(10))
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    if st.checkbox("Show statistical summary"):
        st.dataframe(df.describe())
    
    # Visualization Section
    st.subheader("📊 Data Visualizations")
    
    viz_type = st.selectbox(
        "Choose Visualization Type",
        ["Price Distribution", "Categorical Analysis", "Correlation Analysis", "City-wise Analysis"]
    )
    
    if viz_type == "Price Distribution":
        price_distribution_viz(df)
    elif viz_type == "Categorical Analysis":
        categorical_analysis_viz(df)
    elif viz_type == "Correlation Analysis":
        correlation_analysis_viz(df)
    elif viz_type == "City-wise Analysis":
        city_analysis_viz(df)

def price_distribution_viz(df):
    st.subheader("💰 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['Price'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of House Prices')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Price by bedrooms
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='Price', by='Bedrooms', ax=ax)
        ax.set_xlabel('Number of Bedrooms')
        ax.set_ylabel('Price (₹)')
        ax.set_title('Price Distribution by Bedrooms')
        plt.suptitle('')  # Remove default title
        st.pyplot(fig)

def categorical_analysis_viz(df):
    st.subheader("🏷️ Categorical Variables Analysis")
    
    # Get categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    selected_cat = st.selectbox("Select Categorical Variable", cat_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Countplot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create countplot with colors
        counts = df[selected_cat].value_counts()
        bars = ax.bar(counts.index, counts.values, color='lightcoral', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel(selected_cat)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {selected_cat}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Price by category
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='Price', by=selected_cat, ax=ax)
        ax.set_xlabel(selected_cat)
        ax.set_ylabel('Price (₹)')
        ax.set_title(f'Price Distribution by {selected_cat}')
        plt.suptitle('')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def correlation_analysis_viz(df):
    st.subheader("🔗 Correlation Analysis")
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    st.pyplot(fig)
    
    # Feature correlation with price
    st.subheader("📊 Feature Correlation with Price")
    price_corr = df[numeric_cols].corr()['Price'].sort_values(ascending=False)
    price_corr = price_corr.drop('Price')  # Remove self-correlation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in price_corr.values]
    bars = ax.barh(price_corr.index, price_corr.values, color=colors, alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.3f}',
               ha='left' if width > 0 else 'right',
               va='center', fontweight='bold')
    
    ax.set_xlabel('Correlation with Price')
    ax.set_title('Feature Correlation with House Price')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    st.pyplot(fig)

def city_analysis_viz(df):
    st.subheader("🏙️ City-wise Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average price by city
        city_price = df.groupby('City')['Price'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(city_price.index, city_price.values, 
                     color='lightgreen', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'₹{int(height/1000)}K',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('City')
        ax.set_ylabel('Average Price (₹)')
        ax.set_title('Average House Price by City')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Number of properties by city
        city_count = df['City'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(city_count.index, city_count.values,
                     color='lightblue', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('City')
        ax.set_ylabel('Number of Properties')
        ax.set_title('Property Count by City')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# PART 2: House Price Prediction
def prediction_section(df):
    st.header("🎯 House Price Prediction")
    
    # Prepare data for modeling
    X = df.drop(['Price'], axis=1)
    y = df['Price']
    
    # Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # sidebar model selection
    st.header("⚙️ Model Selection")
    model_choice=st.selectbox(
        "Select Model",
        ("Linear Regression", "Random Forest", "Gradient Boosting")
    )

    # set parameter grids
    if model_choice=='Linear Regression': 
        model=LinearRegression()
        param_grid={}
    elif model_choice=='Random Forest':
        model=RandomForestRegressor(random_state=42)
        param_grid={
            'n_estimators':[50, 100, 200],
            'max_depth':[None, 5, 10]
        }
    else:
        model=GradientBoostingRegressor(random_state=42)
        param_grid={
            'n_estimators':[50, 100, 200],
            'learning_rate':[0.01, 0.1, 0.2],
            'max_depth':[3, 5, 7]
        }

    # Hyperparameter tuning if params exist
    if param_grid:
        grid=GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model=grid.best_estimator_
        st.success(f"Best Params: {grid.best_params_}")
    else:
        best_model=model
        best_model.fit(X_train, y_train)

    
    # Model performance
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("RMSE", f"₹{rmse:,.0f}")
    
    # Input section for prediction
    st.subheader("🏠 Predict House Price")
    st.write("Enter the house details to get price prediction:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.slider("Area (sq ft)", int(df['Area'].min()), int(df['Area'].max()), int(df['Area'].mean()))
        bedrooms = st.selectbox("Bedrooms", sorted(df['Bedrooms'].unique()))
        bathrooms = st.selectbox("Bathrooms", sorted(df['Bathrooms'].unique()))
        stories = st.selectbox("Stories", sorted(df['Stories'].unique()))
    
    with col2:
        parking = st.selectbox("Parking", sorted(df['Parking'].unique()))
        age = st.slider("Age (years)", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
        city = st.selectbox("City", df['City'].unique())
        furnishing = st.selectbox("Furnishing", df['Furnishing'].unique())
    
    with col3:
        main_road = st.selectbox("Main Road", df['Main Road'].unique())
        guest_room = st.selectbox("Guest Room", df['Guest Room'].unique())
        basement = st.selectbox("Basement", df['Basement'].unique())
        water_supply = st.selectbox("Water Supply", df['Water Supply'].unique())
        air_conditioning = st.selectbox("Air Conditioning", df['Air Conditioning'].unique())
        preferred_tenant = st.selectbox("Preferred Tenant", df['Preferred Tenant'].unique())
        locality_rating = st.slider("Locality Rating", 1, 10, 5)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Stories': [stories],
        'Parking': [parking],
        'Age': [age],
        'City': [city],
        'Furnishing': [furnishing],
        'Main Road': [main_road],
        'Guest Room': [guest_room],
        'Basement': [basement],
        'Water Supply': [water_supply],
        'Air Conditioning': [air_conditioning],
        'Preferred Tenant': [preferred_tenant],
        'Locality Rating': [locality_rating]
    })
    
    # Encode input data
    input_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Align columns with training data
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]
    
    # Make prediction
    if st.button("🔮 Predict Price", type="primary"):
        prediction = best_model.predict(input_encoded)[0]
        
        st.success(f"Predicted House Price: ₹{prediction:,.0f}")
        
if __name__ == "__main__":
    main()
