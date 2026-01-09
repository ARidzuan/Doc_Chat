import os
import chainlit as cl

def load_users():
    users = {}
    raw = os.getenv("AUTH_USERS", "")
    for pair in raw.split(","):
        if not pair:
            continue
        username, password = pair.split(":")
        users[username] = password
    return users

USERS = load_users()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if USERS.get(username) == password:
        return cl.User(
            identifier=username,
            metadata={"role": "internal"}
        )
    return None
