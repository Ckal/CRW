#####################################
##   
#####################################

from langchain_community.llms import HuggingFaceHub

#from langchain_community.llms import HuggingFaceHub

llm_zephyr_7b_beta = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 1000,
        "top_k": 2,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools.tools import SerperDevTool

#from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  max_execution_time=120,
  max_iter=50,  # Optional
  max_rpm=25, # Optional
  llm=llm_zephyr_7b_beta
  # You can pass an optional llm attribute specifying what mode you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
  #
  # import os
  # os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
  #
  # OR
  #
  # from langchain_openai import ChatOpenAI
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
)

writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  max_execution_time=120,
  max_iter=50,  # Optional
  max_rpm=25, # Optional
  allow_delegation=False,
  llm=llm_zephyr_7b_beta
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=1, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
#result = crew.kickoff()

#print("######################")
#print(result)

##################
###### other models:
# "Trelis/Llama-2-7b-chat-hf-sharded-bf16"
# "bn22/Mistral-7B-Instruct-v0.1-sharded"
# "HuggingFaceH4/zephyr-7b-beta"

# function for loading 4-bit quantized model
def load_model( ):

    model =  HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"max_length": 1048, "temperature":0.2, "max_new_tokens":256, "top_p":0.95, "repetition_penalty":1.0},
    )
    
    return model

###############
#####
#####
#####
####
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# middlewares to allow cross orgin communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], 
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*'],
)


@app.post("/generate/")
def generate(user_input, history=[]):
    print("######################")
    
    result = crew.kickoff()
    print("######################")
    print(result)
    return result

# load the model asynchronously on startup and save it into memory 
@app.on_event("startup")
async def startup():
    # Get your crew to work!
   # result = crew.kickoff()

    print("######################")
   # print(result)
   