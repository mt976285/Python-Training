import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# --- FIX: Use modern Pydantic import ---
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Define the tools the agent will use ---
# Tools allow the agent to interact with its environment.
# These tools are simplified examples; a real-world agent would use more robust versions.

class FileExplorerToolInput(BaseModel):
    path: str = Field(description="The file path to read or find.")

def read_file_tool(path: str) -> str:
    """Reads and returns the content of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: Could not read file at {path}. {e}"

def list_directory_tool(path: str) -> str:
    """Lists the files and directories at a given path."""
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"Error: Could not list directory at {path}. {e}"

def execute_python_code_tool(code: str) -> str:
    """Executes Python code in a controlled environment and returns the output."""
    try:
        # A safer version would use a sandboxed environment like E2B
        # This is a very simple and unsafe local execution.
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get('result', 'Execution successful (no explicit result)'))
    except Exception as e:
        return f"Execution failed: {e}"

# Add more sophisticated tools as needed, such as:
# - A Git tool to access repo history
# - A search tool (like SerperDevTool) for external documentation
# - A logging tool to analyze specific log files (e.g., from a web server)

# Create the LangChain Tools
read_file = Tool(name="read_file", func=read_file_tool, description="Reads content from a specified file path.")
list_dir = Tool(name="list_directory", func=list_directory_tool, description="Lists contents of a directory.")
run_python = Tool(name="execute_python", func=execute_python_code_tool, description="Executes Python code. Returns the output or any error.")

# --- 2. Create the agent ---
# The agent uses an LLM as its reasoning engine and can use the tools.

llm = ChatOpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY, temperature=0)
tools = [read_file, list_dir, run_python]

# Define the agent's prompt
# This is crucial for guiding the agent's behavior
prompt_template = """
You are an AI debugging assistant. Your goal is to analyze log files, examine code,
and propose a solution to a software issue.

You have access to the following tools:
{tools}

Use the following thought process to solve the problem:
- **Thought**: What should I do next?
- **Action**: Use a tool. The action and its input must be a single string containing a JSON object, like this: {{ "action": "tool_name", "action_input": "tool_input" }}
- **Observation**: The result of the action.

Begin!

Problem: Analyze the logs to find the cause of the issue, debug the code in the 'app/' directory, and provide a fix.

Begin.

Thought: I need to start by examining the logs to identify any errors or warnings. Then I will use that information to investigate the code.
Action: {{"action": "list_directory", "action_input": "logs"}}
"""

# Create the agent using LangGraph's prebuilt ReAct agent
# This handles the Plan-Execute-Observe loop
agent_runnable = create_react_agent(llm, tools,  prompt=prompt_template)

# --- 3. Run the agent and observe the output ---
# To make this example runnable, we simulate a file system.
def setup_simulated_repo():
    """Sets up a mock code repository with a logs directory."""
    if os.path.exists('simulated_repo'):
        shutil.rmtree('simulated_repo')
    
    Path('simulated_repo/app').mkdir(parents=True, exist_ok=True)
    Path('simulated_repo/logs').mkdir(parents=True, exist_ok=True)
    
    with open('simulated_repo/app/main.py', 'w') as f:
        f.write("""
def divide_by_zero():
    return 10 / 0

if __name__ == "__main__":
    result = divide_by_zero()
    print(result)
""")
    
    with open('simulated_repo/logs/error.log', 'w') as f:
        f.write("INFO: Application started.\n")
        f.write("ERROR: division by zero: A critical error occurred.\n")
        f.write("INFO: Application stopped unexpectedly.\n")

    os.chdir('simulated_repo')

def run_agent():
    """Runs the LangChain agent and prints the output."""
    setup_simulated_repo()

    print("--- Starting Agent Execution ---\n")
    
    # LangGraph requires a specific format for its input
    agent_input = {"input": "Analyze the error log, debug the 'app/main.py' file, and propose a fix."}
    
    # Running the agent
    for chunk in agent_runnable.stream(agent_input):
        for key, value in chunk.items():
            print(f"**Agent State ({key}):**\n{value}")
        print("\n---\n")

    os.chdir('..')

if __name__ == "__main__":
    run_agent()

