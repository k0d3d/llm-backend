import os


from crewai import LLM


from app.core.types import LLMProvider, SelectedLLM

# get the value for the selected model from the environment variable for basic and turbo
USE_BASIC = os.getenv("USE_BASIC")
USE_TURBO = os.getenv("USE_TURBO")
TOGETHER_AI_API_TOKEN = os.getenv("TOGETHER_AI_API_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# open_ai_basic = ChatOpenAI(model="gpt-5-mini", temperature=0.8)
open_ai_basic = LLM(
    model="gpt-5-nano",
)
open_ai_turbo = LLM(
    model="gpt-5-mini",
)

together_basic = LLM(
    model="together_ai/togethercomputer/llama-2-70b-chat",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=TOGETHER_AI_API_TOKEN,
)

together_turbo = LLM(
    model="together_ai/togethercomputer/llama-2-70b",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=TOGETHER_AI_API_TOKEN,
)

replicate_basic = LLM(
    model="replicate/meta/meta-llama-3-8b-instruct",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=REPLICATE_API_TOKEN,
)

replicate_turbo = LLM(
    model="replicate/meta/meta-llama-3-70b-instruct",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=REPLICATE_API_TOKEN,
)

claude_basic = LLM(
    model="claude-3-haiku-20240307",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=ANTHROPIC_API_KEY,
)
claude_turbo = LLM(
    model="claude-3-5-sonnet-20240620",
    temperature=0.7,
    top_k=1,
    repetition_penalty=1.0,
    api_key=ANTHROPIC_API_KEY,
)

selected_model = {
    "OPEN_AI": (open_ai_basic, open_ai_turbo),
    "TOGETHER": (together_basic, together_turbo),
    "REPLICATE": (replicate_basic, replicate_turbo),
    "CLAUDE": (claude_basic, claude_turbo),
}


def llm_switcher(llm: SelectedLLM):
    provider = llm["provider"]
    name = llm["name"]

    if LLMProvider.OPENAI == provider:
        return LLM(
            model=name,
        )

    if LLMProvider.CLAUDE == provider:
        return LLM(
            model=name,
            temperature=0.7,
            api_key=ANTHROPIC_API_KEY
        )

    if LLMProvider.GOOGLE == provider:
        return LLM(
            model=name,
            temperature=0.7,
            api_key=GEMINI_API_KEY
        )

    if LLMProvider.REPLICATE == provider:
        return LLM(
            model=f"replicate/{name}",
            temperature=0.7,
            stop="END",
            api_key=REPLICATE_API_TOKEN
        )

    if LLMProvider.TOGETHER_AI == provider:
        return LLM(
            model=f"together_ai/{name}",
            temperature=0.7,
            api_key=TOGETHER_AI_API_TOKEN,
            stop=["<|eot_id|>","<|eom_id|>"],

        )

    return open_ai_basic


def current_basic_model(selected_llm: SelectedLLM = None):
    return (
        llm_switcher(selected_llm)
        if selected_llm
        else selected_model[USE_BASIC][1]
        if USE_BASIC in selected_model
        else open_ai_turbo
    )


def current_turbo_model():
    return (
        selected_model[USE_TURBO][1] if USE_TURBO in selected_model else open_ai_turbo
    )


def get_available_llm_model():
    llm_list = [
        # {
        #     "name": "o3",
        #     "provider": LLMProvider.OPENAI,
        #     "kargs": {},
        # },
        {
            "name": "gpt-5-mini",
            "provider": LLMProvider.OPENAI,
            "kargs": {},
        },
        {
            "name": "gpt-5-nano",
            "provider": LLMProvider.OPENAI,
            "kargs": {},
        },
        {
            "name": "gpt-5",
            "provider": LLMProvider.OPENAI,
            "kargs": {},
        },
        # {
        #     "name": "gpt-5-mini",
        #     "provider": LLMProvider.OPENAI,
        #     "kargs": {},
        # },
        {
            "name": "gemini/gemini-2.5-flash-preview-04-17",
            "provider": LLMProvider.GOOGLE,
            "kargs": {},
        },
        {
            "name": "gemini/gemini-2.0-flash-exp",
            "provider": LLMProvider.GOOGLE,
            "kargs": {},
        },
        {
            "name": "anthropic/claude-3.5-sonnet",
            "provider": LLMProvider.REPLICATE,
            "kargs": {},
        },
        {
            "name": "anthropic/claude-3.7-sonnet",
            "provider": LLMProvider.REPLICATE,
            "kargs": {},
        },
        {
            "name": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "provider": LLMProvider.TOGETHER_AI,
            "kargs": {},
        },
        {
            "name": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "provider": LLMProvider.TOGETHER_AI,
            "kargs": {},
        },
        {
            "name": "deepseek-ai/DeepSeek-R1",
            "provider": LLMProvider.TOGETHER_AI,
            "kargs": {},
        },
        {
            "name": "deepseek-ai/DeepSeek-V3",
            "provider": LLMProvider.TOGETHER_AI,
            "kargs": {},
        },
        {
            "name": "Qwen/Qwen3-235B-A22B-fp8",
            "provider": LLMProvider.TOGETHER_AI,
            "kargs": {},
        },
        # {
        #     "name": "claude-3-haiku-20240307",
        #     "provider": LLMProvider.CLAUDE,
        #     "kargs": {},
        # },
        # {
        #     "name": "claude-3-5-sonnet-20240620",
        #     "provider": LLMProvider.CLAUDE,
        #     "kargs": {},
        # },
    ]

    return llm_list
