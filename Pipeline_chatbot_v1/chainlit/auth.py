import os
import chainlit as cl
from pathlib import Path

def get_env_file_path():
    """Get the path to the .env file"""
    return Path(__file__).parent / ".env"

def load_users():
    """Load users from AUTH_USERS environment variable"""
    users = {}
    raw = os.getenv("AUTH_USERS", "")
    for pair in raw.split(","):
        if not pair:
            continue
        if ":" not in pair:
            continue
        username, password = pair.split(":", 1)
        users[username] = password
    return users

def reload_users():
    """Reload users from .env file by re-reading the environment"""
    global USERS
    env_file = get_env_file_path()
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("AUTH_USERS="):
                    # Extract the value after AUTH_USERS=
                    auth_users_value = line.split("=", 1)[1]
                    os.environ["AUTH_USERS"] = auth_users_value
                    break
    USERS = load_users()

def add_user_to_env(username: str, password: str) -> tuple[bool, str]:
    """
    Add a new user to the .env file
    Returns (success, message)
    """
    # Validate username and password
    if not username or not password:
        return False, "Username and password cannot be empty"

    if ":" in username or "," in username:
        return False, "Username cannot contain ':' or ','"

    if ":" in password or "," in password:
        return False, "Password cannot contain ':' or ','"

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if len(password) < 6:
        return False, "Password must be at least 6 characters"

    # Check if username already exists
    if username in USERS:
        return False, "Username already exists"

    # Read the .env file
    env_file = get_env_file_path()
    if not env_file.exists():
        return False, ".env file not found"

    with open(env_file, 'r') as f:
        lines = f.readlines()

    # Find and update the AUTH_USERS line
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith("AUTH_USERS="):
            # Extract existing value
            current_value = line.strip().split("=", 1)[1]
            # Append new user
            new_value = f"{current_value},{username}:{password}"
            lines[i] = f"AUTH_USERS={new_value}\n"
            updated = True
            break

    if not updated:
        return False, "AUTH_USERS not found in .env file"

    # Write back to .env file
    try:
        with open(env_file, 'w') as f:
            f.writelines(lines)

        # Reload users into memory
        reload_users()
        return True, "User registered successfully! You can now login."
    except Exception as e:
        return False, f"Error writing to .env file: {str(e)}"

USERS = load_users()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Handle login authentication"""
    # Reload users to pick up any new sign-ups
    reload_users()

    if USERS.get(username) == password:
        return cl.User(
            identifier=username,
            metadata={"role": "internal"}
        )
    return None
