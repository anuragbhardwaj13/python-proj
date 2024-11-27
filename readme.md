# Create and activate virtual environment
python3 -m venv hospital_env
source hospital_env/bin/activate

# Install required packages
pip install streamlit pandas scikit-learn joblib

# Run the app
streamlit run app.py