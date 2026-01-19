This is an application meant for Senior Citizens who need help with navigating and performing tasks on their computer. 
This application is meant to automate and perform day-to-day tasks for users.

Uses a multi-agentic workflow to understand user queries and translate that into actual physical tasks. 


2 LLMs (3 potentially):

#1: Router LLM
- Understands the prompt
- Classifies what the user wants (which domains are involved)
- Assesses risk (low / medium / high)
- Decides if clarification is needed before doing anything
- Guides the next step (which planner prompt + policies to apply)
- Also reasses its output if there are any issues (with Pydantic)

#2: Planner LLM:
- Plans the steps in which tasks need to be made
- Returns a JSON output with a list of all the tools that need to be called
- 