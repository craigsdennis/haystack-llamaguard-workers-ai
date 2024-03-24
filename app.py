from dotenv import load_dotenv
from haystack import Pipeline
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
    st.session_state.messages = [] # ChatMessage.from_system("You always respond with 'I can help with that!' and nothing else")]

# Display all messages if they exist?
for msg in st.session_state.messages:
    with st.chat_message(msg.role.value):
        st.markdown(msg.content)


if prompt := st.chat_input("Let's chat"):
    st.session_state.messages.append(ChatMessage.from_user(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    results = pipeline.run({"user_moderator": {"messages": st.session_state.messages}})
    if "busted" in results:
        msg = results["busted"]["response"]
    else:
        if "assistant_moderator" in results:
            msg = results["assistant_moderator"]["safe_messages"][-1]
    st.session_state.messages.append(msg)
    with st.chat_message("assistant"):
        st.markdown(msg.content)
