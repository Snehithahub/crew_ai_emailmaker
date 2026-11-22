from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from crewai.tools import BaseTool, tool
load_dotenv()

llm = LLM(model="groq/openai/gpt-oss-120b", temperature=0)

user_text = "we have made changes for OBM,u can test now LOL"

@tool("shortcut to fullform")
def shortcuts(email):
    """Expands chat short forms and abbreviations (AFAIK, LOL, OBM) in a message.
    Returns the message with every short form replaced."""
    suggestions=[]
    replacement={"AFAIK":"as far as i know","LOL":"laugh out loud","OBM":"obligation management usecase"}
    for i,j in replacement.items():
            if i in email:
                suggestions.append(f"replace {i} with {j}")
        
    return ",".join(suggestions) if suggestions else "no suggestions,can continue"   
    
    
agent_assistant = Agent(
    role="email assistant agent",
    goal="improve email into proper format.",
    backstory="you are an expert in communication through email",
    llm=llm,
    tools=[shortcuts],
    verbose=True,
)

friendly_assistant = Agent(
    role="email assistant agent",
    goal="improve email into proper format.",
    backstory="you are an expert in communication through email",
    llm=llm,
    tools=[shortcuts],
    verbose=True,
)

editor=Agent(role="email editor",
             goal="to edit in a best polished way",
             backstory="Senior communications lead with an eye for tone and brevity.",
             llm=llm,
             tools=[shortcuts],
             verbose=True)


task_agent = Task(
    agent=agent_assistant,
    description="your task is to write {input_to_agent} into proper email format",
    expected_output="into proper email format only in polite way",
    async_execution=True
)

friendly_task = Task(
    agent=friendly_assistant,
    description="Write a FRIENDLY but professional email for this message: {input_to_agent}",
    expected_output="Subject line, greeting, body, sign-off. Only the email.",
    async_execution=True       
)


editor_task=Task(agent=editor,
                 description="compare both and get the best mail",
                 expected_output="One final email: subject line, greeting, body, sign-off. No commentary.",
                 context=[friendly_task,task_agent])

crew = Crew(agents=[agent_assistant,editor,friendly_assistant], tasks=[task_agent,friendly_task,editor_task])

if __name__ == "__main__":
    result = crew.kickoff(inputs={"input_to_agent": user_text})
    print("\n===== FINAL EMAIL =====\n")
    print(result)
