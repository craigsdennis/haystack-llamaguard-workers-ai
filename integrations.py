import dataclasses

from haystack import component
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret
import requests
from requests.adapters import HTTPAdapter
from typing import List, Dict, Optional

# TODO: Handle Streaming


class BaseCloudflareGenerator:

    def __init__(
        self,
        account_id: Secret = Secret.from_env_var("CLOUDFLARE_ACCOUNT_ID"),
        api_token: Secret = Secret.from_env_var("CLOUDFLARE_API_TOKEN"),
        model: str = "@cf/meta/llama-2-7b-chat-int8",
    ):
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id.resolve_value()}/ai/run/{model}"
        self.headers = {"Authorization": f"Bearer {api_token.resolve_value()}"}

    def _convert_to_message_json(
        self, messages: List[ChatMessage]
    ) -> List[Dict[str, str]]:
        valid_keys = {"role", "content"}
        converted = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            filtered_message = {
                k: v for k, v in message_dict.items() if k in valid_keys and v
            }
            converted.append(filtered_message)
        return converted


@component
class CloudflareChatGenerator(BaseCloudflareGenerator):

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        # TODO: Handle streaming
        s = requests.Session()
        s.mount(self.url, HTTPAdapter(max_retries=3))
        response = s.post(
            self.url,
            headers=self.headers,
            json={"messages": self._convert_to_message_json(messages)},
        )
        json_value = response.json()
        messages.append(ChatMessage.from_assistant(json_value["result"]["response"]))
        return {"replies": messages}


@component
class LlamaGuard(BaseCloudflareGenerator):

    DEFAULT_UNSAFE_CATEGORIES = """
01: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
-  Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
02: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
03: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
04: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
05: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
06: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
"""

    def __init__(
        self,
        account_id: Secret = Secret.from_env_var("CLOUDFLARE_ACCOUNT_ID"),
        api_token: Secret = Secret.from_env_var("CLOUDFLARE_API_TOKEN"),
        model: str = "@hf/thebloke/llamaguard-7b-awq",
        unsafe_categories: str = None,
    ):
        if unsafe_categories is None:
            unsafe_categories = self.DEFAULT_UNSAFE_CATEGORIES
        self.unsafe_categories = unsafe_categories
        super(LlamaGuard, self).__init__(account_id, api_token, model)

    def unsafe_reasoning_from_response(self, response):
        parts = response.split("\n")
        if len(parts) < 2:
            return []
        # We asked for comma separated
        codes = parts[1].split(",")
        reasons = []
        lines = self.unsafe_categories.split("\n")
        for code in codes:
            # Sometimes capital O is returned instead of 0
            code = code.replace("O", "0")
            for line in lines:
                if line.startswith(code):
                    reasons.append(line)
        return reasons

    @component.output_types(
        safe_response=Optional[str],
        unsafe_response=Optional[str],
        safe_messages=List[ChatMessage],
        unsafe_messages=List[ChatMessage],
        reasons=List[str],
    )
    def run(self, messages: List[ChatMessage]):
        # Pull out the latest message
        role_title = messages[-1].role.value.title()

        # User: It looks like this
        #
        # Assistant: Thank you for explaining
        #
        conversation_string = "\n\n".join(
            [
                f"{message.role.value.title()}: {message.content}\n\n"
                for message in messages
                if message.role.value in {"user", "assistant"}
            ]
        )
        outer_prompt = f"""Task: Check if there is unsafe content in '{role_title}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self.unsafe_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
{conversation_string}
<END CONVERSATION>

Provide your safety assessment for '{role_title}' in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of all violated categories.
        """
        s = requests.Session()
        s.mount(self.url, HTTPAdapter(max_retries=3))
        response = s.post(
            self.url, headers=self.headers, json={"prompt": outer_prompt}
        )
        json_value = response.json()
        response = json_value["result"]["response"].strip()
        if response.startswith("unsafe"): # DEMO: and role_title == "Assistant":
            reasons = self.unsafe_reasoning_from_response(response)
            return {
                "unsafe_response": response,
                "reasons": reasons,
                "unsafe_messages": messages,
            }
        else:
            return {"safe_response": response, "safe_messages": messages}


@component
class BustedGenerator:

    @component.output_types(response=ChatMessage)
    def run(self, user_reasons: List[str] = None, assistant_reasons: List[str] = None):
        print({"user_reasons": user_reasons, "assistant_reasons": assistant_reasons})
        if user_reasons:
            return {
                "response": ChatMessage.from_assistant(
                    f"You said something naughty: {user_reasons}"
                )
            }
        elif assistant_reasons:
            return {
                "response": ChatMessage.from_assistant(
                    f"The generated answer was naughty: {assistant_reasons}"
                )
            }
