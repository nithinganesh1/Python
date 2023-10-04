# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:46:28 2023

@author: Nithin
"""
import streamlit as st

st.sidebar.header("Agent Authentication")
agent_username = st.sidebar.text_input("Username")
agent_password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

#main
st.title("Wonder Airlines Agent Interface")

if login_button:
    st.components.v1.iframe(
        src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FMQzyrOxt3P8EI58NJeZCbT%2FWonder-Airlines%3Ftype%3Dwhiteboard%26node-id%3D0%253A1%26t%3D2siwyaQfY2i2eSUB-1",
        width=800,
        height=450,
    )

    if agent_username == "nithin" and agent_password == "nithin123":
        st.sidebar.success("Logged in as " + agent_username)
        st.write("Welcome, " + agent_username + "!")

        st.header("Notes")
        st.text_area("Type Notes here:")

        st.header("Request Assistance from AI")
        ai_assistance = st.button("Request AI Assistance")

        if ai_assistance:
            st.success("AI assistance requested. Wait for AI response...")

    else:
        st.sidebar.error("Authentication failed. Please check your username and password.")
else:
    st.sidebar.warning("Please log in to access the agent interface.")


