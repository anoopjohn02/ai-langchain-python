from langchain.tools import Tool

def run_greeting(name):
    print(name)
    return name + " from AI! "

run_greeting_tool = Tool.from_function(
    name="run_greeting",
    description="Run a greeting.",
    func=run_greeting
)