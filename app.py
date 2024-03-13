import os
from typing import Dict, List

from dotenv import load_dotenv
from haystack import component, Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
import streamlit as st

from integrations import CloudflareChatGenerator, LlamaGuard


load_dotenv()


@component
class BustedGenerator:
    
    def __init__(self, moderator):
        self.moderator = moderator

    @component.output_types(response=ChatMessage)
    def run(self, response: str):
        reasons = self.moderator.unsafe_reasoning_from_response(response)
        return {"response": ChatMessage.from_assistant(f"You naughty: {reasons}")}


llm = CloudflareChatGenerator(
    account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
    api_token=os.environ["CLOUDFLARE_API_TOKEN"],
    model="@cf/meta/llama-2-7b-chat-int8",
)

moderator = LlamaGuard(
    account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
    api_token=os.environ["CLOUDFLARE_API_TOKEN"],
)

routes = [
    {
        "condition": "{{response.startswith('unsafe')}}",
        "output": "{{response}}",
        "output_name": "unsafe_query",
        "output_type": str,
    },
    {
        "condition": "{{response.startswith('safe')}}",
        "output": "{{messages}}",
        "output_name": "safe_query",
        "output_type": List[ChatMessage],
    },
]

router = ConditionalRouter(routes=routes)

pipeline = Pipeline()
pipeline.add_component("moderator", instance=moderator)
pipeline.add_component("router", router)
pipeline.add_component("busted", instance=BustedGenerator(moderator))
pipeline.add_component("llm", instance=llm)
pipeline.connect("moderator", "router")
pipeline.connect("router.unsafe_query", "busted")
pipeline.connect("router.safe_query", "llm.messages")


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
    results = pipeline.run({"messages": st.session_state.messages})
    with st.chat_message("assistant"):
        if "llm" in results:
            msg = results["llm"]["response"]
        else:
            msg = results["busted"]["response"]
        st.markdown(msg.content)
    st.session_state.messages.append(msg)
