import os
from typing import Dict, List

from dotenv import load_dotenv
from haystack import component, Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
import streamlit as st

from integrations import CloudflareChatGenerator, LlamaGuard, BustedGenerator


load_dotenv()

llm = CloudflareChatGenerator()

user_moderator = LlamaGuard()

assistant_moderator = LlamaGuard()

pipeline = Pipeline()
pipeline.add_component("user_moderator", instance=user_moderator)
pipeline.add_component("busted", instance=BustedGenerator())
pipeline.add_component("llm", instance=llm)
pipeline.add_component("assistant_moderator", instance=assistant_moderator)

pipeline.connect("user_moderator.safe_messages", "llm.messages")
pipeline.connect("user_moderator.reasons", "busted.user_reasons")
pipeline.connect("llm.replies", "assistant_moderator.messages")
pipeline.connect("assistant_moderator.reasons", "busted.assistant_reasons")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display them if they exist?

if prompt := st.chat_input("Let's chat"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(ChatMessage.from_user(prompt))
    results = pipeline.run({"user_moderator": {"messages": st.session_state.messages}})
    with st.chat_message("assistant"):
        if "assistant_moderator" in results:
            if "busted" in results:
                msg = results["assistant_moderator"]["unsafe_messages"][-1]
                st.markdown(results["busted"]["response"].content)
            else:
                msg = results["assistant_moderator"]["safe_messages"][-1]
            st.markdown(msg.content)
            st.session_state.messages.append(msg)
        elif "user_moderator" in results:
            if "busted" in results:
                st.markdown(results["busted"]["response"].content)