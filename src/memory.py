user_sessions = {}

def add_message_to_history(user_id: str, user_message: str, bot_response: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    user_sessions[user_id].append((user_message, bot_response))

def get_conversation_history(user_id: str):
    return user_sessions.get(user_id, [])
