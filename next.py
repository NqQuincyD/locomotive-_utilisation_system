import streamlit as st
import sqlite3
import re
import bcrypt

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, is_admin INTEGER)''')
    conn.commit()
    conn.close()

# Check if a user exists
def user_exists(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user is not None

# Add a new user
def add_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
              (username, hashed_password, is_admin))
    conn.commit()
    conn.close()

# Verify user credentials
def verify_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND is_admin=?", (username, is_admin))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        return True
    return False

# Check password strength
def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# Initialize the database
init_db()

# Streamlit app
st.title("User Authentication System")

# Initialize session state for admin verification
if 'admin_verified' not in st.session_state:
    st.session_state.admin_verified = False

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Login", "Sign Up"])

if page == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    user_type = st.radio("Login as:", ("User", "Admin"))
    
    if st.button("Login"):
        is_admin = 1 if user_type == "Admin" else 0
        if verify_user(username, password, is_admin):
            st.success(f"Logged in successfully as {user_type}")
        else:
            st.error("Invalid username or password")

elif page == "Sign Up":
    st.header("Sign Up")
    
    if not st.session_state.admin_verified:
        # Admin verification input
        admin_username = st.text_input("Admin Username (to access Sign Up)", placeholder="Enter admin username")
        admin_password = st.text_input("Admin Password", type="password")
        
        if st.button("Verify Admin"):
            if verify_user(admin_username, admin_password, is_admin=1):
                st.session_state.admin_verified = True
                st.success("Admin verified. You can now sign up users.")
            else:
                st.error("Admin verification failed.")
    else:
        # Show user creation form
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        user_type = st.radio("User Type:", ("User", "Admin"))
        
        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not is_strong_password(new_password):
                st.error("Password is not strong enough. It should have at least 8 characters, including uppercase and lowercase letters, numbers, and special characters.")
            elif user_exists(new_username):
                st.error("Username already exists")
            else:
                is_admin = 1 if user_type == "Admin" else 0
                add_user(new_username, new_password, is_admin)
                st.success("User created successfully")
                # Keep the sign-up form visible
                new_username = st.text_input("New Username")  # Clear input for new user
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")

# Add some CSS to improve the app's appearance
st.markdown("""
<style>
    .stRadio > label {
        font-weight: bold;
        color: #4A4A4A;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)