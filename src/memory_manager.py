# memory_manager.py

from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

# Creamos un diccionario en session_state para guardar memorias por usuario
def get_memory_for_user(user_id: str):
    if "user_memories" not in st.session_state:
        st.session_state.user_memories = {}

    if user_id not in st.session_state.user_memories:
        st.session_state.user_memories[user_id] = MemorySaver()

    return st.session_state.user_memories[user_id]

