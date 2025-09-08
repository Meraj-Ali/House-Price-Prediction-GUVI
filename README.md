# 🏠 House Price Prediction & Analysis

A Streamlit web app for exploring house price datasets, performing interactive data analysis, and predicting house prices using machine learning models.

## ✨ Features
### 📊 Data Analysis & Visualization

- Dataset overview with key stats (records, features, average price, price range).

- Interactive raw data preview and statistical summary.

- Multiple visualizations:

  - Price Distribution (histogram, boxplot by bedrooms)

  - Categorical Analysis (count plots, price by category)

  - Correlation Analysis (heatmap, feature correlation with price)

  - City-wise Analysis (average price by city, property count by city)

### 🎯 House Price Prediction

- Machine Learning models with hyperparameter tuning:

  - Linear Regression

  - Random Forest Regressor

  - Gradient Boosting Regressor

- Model performance metrics:

  - R² Score

  - RMSE (Root Mean Squared Error)

- Interactive prediction form where users input house features to predict the price.

### 🚀 Installation
1️⃣ **Clone the repository**
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

2️⃣ **Create a virtual environment**

Use Python 3.11 or 3.12 (⚠️ some libraries may not work with Python 3.13 yet).

python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

3️⃣ **Install dependencies**
pip install --upgrade pip setuptools wheel cython
pip install -r requirements.txt


If you don’t have a requirements.txt yet, create one with:

pip freeze > requirements.txt

4️⃣ **Run the app**
streamlit run app.py

📂 Project Structure
├── app.py                          # Main Streamlit application
├── enhanced_house_price_dataset.csv # Dataset file
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation

📊 Dataset

The app expects a CSV file named enhanced_house_price_dataset.csv with at least the following columns:

Price (target variable)

Area

Bedrooms

Bathrooms

Stories

Parking

Age

City

Furnishing

Main Road

Guest Room

Basement

Water Supply

Air Conditioning

Preferred Tenant

Locality Rating

⚙️ Requirements

Python 3.11 or 3.12

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

scikit-learn

Install via:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn

📸 Screenshots
Home Page

(Insert screenshot of Streamlit app home page here)

Data Analysis Section

(Insert screenshot of visualization page here)

Prediction Section

(Insert screenshot of prediction form and result here)

💡 Future Improvements

Add support for more ML models (e.g., XGBoost, LightGBM).

Deploy on Streamlit Cloud or Heroku.

Add user-uploaded datasets.

Save and compare model results.

👨‍💻 Author

Developed by [Your Name] ✨
