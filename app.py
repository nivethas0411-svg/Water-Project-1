import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ippo Vanthu Ground - Water Waste Management",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #333;
        padding: 0.5rem;
        text-align: center;
    }
    .option-button {
        background-color: #0066cc;
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 18px;
        margin: 10px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .message-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'reports' not in st.session_state:
        st.session_state.reports = []
    if 'comments' not in st.session_state:
        st.session_state.comments = []
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

initialize_session_state()

# User database (in production, use secure database)
USERS = {
    "officer1": {"password": "officer123", "role": "officer", "name": "Officer Kumar"},
    "officer2": {"password": "officer456", "role": "officer", "name": "Officer Priya"},
    "officer3": {"password": "officer789", "role": "officer", "name": "Officer Raj"},
    "admin": {"password": "admin123", "role": "admin", "name": "Admin"},
    "citizen1": {"password": "citizen123", "role": "citizen", "name": "Citizen Muthu"},
    "citizen2": {"password": "citizen456", "role": "citizen", "name": "Citizen Lakshmi"}
}

# Generate sample data for AI model
@st.cache_data
def generate_training_data(n_samples=2000):
    """Generate comprehensive training data for water waste prediction"""
    np.random.seed(42)
    
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'water_level': np.random.uniform(10, 100, n_samples),
        'rainfall': np.random.uniform(0, 50, n_samples),
        'temperature': np.random.uniform(18, 38, n_samples),
        'humidity': np.random.uniform(30, 95, n_samples),
        'population_density': np.random.uniform(50, 250, n_samples),
        'time_of_day': [i % 24 for i in range(n_samples)],
        'day_of_week': [(i // 24) % 7 for i in range(n_samples)],
        'is_weekend': [1 if (i // 24) % 7 >= 5 else 0 for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate water waste with realistic patterns
    df['water_waste'] = (
        0.4 * df['water_level'] + 
        0.25 * df['rainfall'] + 
        0.15 * df['temperature'] + 
        0.08 * df['humidity'] + 
        0.07 * df['population_density'] +
        0.05 * (df['time_of_day'] > 12).astype(int) * 10 +  # Higher waste in afternoon
        np.random.normal(0, 3, n_samples)
    )
    
    # Add seasonal variations
    df['month'] = df['timestamp'].dt.month
    summer_months = [3, 4, 5]
    df.loc[df['month'].isin(summer_months), 'water_waste'] *= 1.3
    
    return df

# Train AI model
@st.cache_resource
def train_water_waste_model():
    """Train Random Forest model for water waste prediction"""
    df = generate_training_data()
    
    features = ['water_level', 'rainfall', 'temperature', 'humidity', 
                'population_density', 'time_of_day', 'day_of_week', 'is_weekend']
    X = df[features]
    y = df['water_waste']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }
    
    return model, metrics, df

# Login page
def login_page():
    st.markdown("<h1 class='main-header'>💧 IPPO VANTHU GROUND</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Water Waste Management System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Romba Decreases Situ Varuthu - Water Level Monitoring & Alert System</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class='info-box'>
                <h3 style='text-align: center;'>🔐 Login to Access System</h3>
                <p style='text-align: center;'>Athoda main reason nirajya yedathula water waste agguthu athukaka people romba stresss</p>
            </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🚪 Login", use_container_width=True):
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_role = USERS[username]["role"]
                    st.session_state.username = USERS[username]["name"]
                    st.session_state.user_id = username
                    st.success(f"✅ Welcome {USERS[username]['name']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        with col_b:
            if st.button("👥 Guest Access", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user_role = "guest"
                st.session_state.username = "Guest User"
                st.rerun()
        
        with col_c:
            if st.button("📝 Sign Up", use_container_width=True):
                st.info("Sign up feature coming soon!")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📋 Demo Credentials - Click to View"):
            st.markdown("""
                **Officers (4 Options):**
                - officer1 / officer123 (Officer Kumar)
                - officer2 / officer456 (Officer Priya)
                - officer3 / officer789 (Officer Raj)
                
                **Admin:**
                - admin / admin123
                
                **Citizens:**
                - citizen1 / citizen123 (Citizen Muthu)
                - citizen2 / citizen456 (Citizen Lakshmi)
                
                **Or use Guest Access**
            """)

# Main application after login
def main_application():
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.user_role.upper()}")
        
        # Notification badge
        unread_notifications = len([n for n in st.session_state.notifications if not n.get('read', False)])
        if unread_notifications > 0:
            st.markdown(f"🔔 **{unread_notifications} New Notifications**")
        
        st.markdown("---")
        
        # Navigation based on role
        if st.session_state.user_role == "citizen":
            page = st.radio(
                "📍 Navigation",
                ["🏠 Home", "⚠️ Report Water Waste", "📨 My Reports", "📊 View Statistics", "🔔 Notifications"]
            )
        elif st.session_state.user_role in ["officer", "admin"]:
            page = st.radio(
                "📍 Navigation",
                ["🏠 Dashboard", "📥 Manage Reports", "🔮 AI Predictions", "📊 Analytics", "💬 Messages", "👥 User Management"]
            )
        else:  # guest
            page = st.radio(
                "📍 Navigation",
                ["🏠 Home", "📊 View Statistics", "ℹ️ About"]
            )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()
        
        # System info
        st.markdown("---")
        st.markdown("###
