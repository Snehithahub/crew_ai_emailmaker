from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from crewai.tools import BaseTool, tool
load_dotenv()

llm = LLM(model="groq/openai/gpt-oss-120b", temperature=0)

user_text = "we have made changes,u can test now"