from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory

from src.config import OPENAI_API_KEY, load_rules
from src.tools import analyzeMarket, placeOrder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2,
                 api_key=OPENAI_API_KEY, max_tokens=1024)

memory = ConversationBufferWindowMemory(k=4, return_messages=True)

SYSTEM_MSG = (
    "You are R0, an autonomous Roostoo trader.\n"
    "Always respond with a JSON tool call like:\n"
    '{"tool":"placeOrder","args":{...}}\n'
    "If no trade is needed, use tool \"none\"."
)

agent = initialize_agent(
    tools=[analyzeMarket, placeOrder],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    max_iterations=6,
    system_message=SYSTEM_MSG,
)

def run_once(config_path="rules.json"):
    rules = load_rules(config_path)
    output = agent.invoke({"input": "Run trade loop", "rules": rules})
    return output["output"]
