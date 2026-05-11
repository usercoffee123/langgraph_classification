"""
LangChain Demo — three core concepts shown end-to-end:

  1. Prompt Templates  — parameterised prompts you can reuse
  2. LLM calls        — talking to an OpenAI model through LangChain
  3. Chains (LCEL)    — piping a prompt → model → output parser together
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads OPENAI_API_KEY from .env


# ── 1. Model ──────────────────────────────────────────────────────────────────
# ChatOpenAI is a thin wrapper around the OpenAI chat-completion API.
# You can swap this for any other LangChain-supported model (Anthropic, Ollama…)
# without changing the rest of your code.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── 2. Prompt Template ────────────────────────────────────────────────────────
# A template keeps your prompt logic separate from your call logic.
# Variables inside {} are filled in at call-time.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains concepts clearly and concisely."),
    ("human", "Explain {concept} in simple terms. Give a one-sentence summary, then a short example."),
])


# ── 3. Chain (LCEL — LangChain Expression Language) ──────────────────────────
# The | operator pipes each component into the next:
#   prompt  →  llm  →  output_parser
# This returns a plain string instead of a message object.
chain = prompt | llm | StrOutputParser()


# ── Run it ────────────────────────────────────────────────────────────────────
def explain(concept: str) -> str:
    return chain.invoke({"concept": concept})


if __name__ == "__main__":
    topics = ["recursion", "neural networks", "REST APIs"]

    for topic in topics:
        print(f"\n{'─' * 60}")
        print(f"Topic: {topic}")
        print("─" * 60)
        print(explain(topic))
