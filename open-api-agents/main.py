
from langchain.chat_models import ChatOpenAI
from langchain.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.greeting import run_greeting_tool
from tools.report import write_report_tool

load_dotenv()

tools=[
    run_query_tool, 
    describe_tables_tool, 
    write_report_tool,
    run_greeting_tool
]
chat = ChatOpenAI()

tables = list_tables()

memory=ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate(
    #input_variables=["content", "messages"],
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executer = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
    memory=memory
)

#agent_executer("How many users are in the database?")
#agent_executer("Good Morning")
#agent_executer("How many users have provided a shipping address?")
agent_executer("Summarize the top 5 most popular products. Write the results to a report file.")

# pip install langchain==0.1.6 langchain-community==0.0.19 langchain-core==0.1.23