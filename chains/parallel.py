from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short questions from {text}',
    input_variables=['text']
)   

prompt3 = PromptTemplate(
    template='Merge the provided notes and questions into a single document:\n\nNotes:\n{notes}\n\nQuestions:\n{questions}  ',
    input_variables=['notes','questions']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model | parser,
        'questions': prompt2 | model2 | parser,
    }
) 

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

final = chain.invoke({'text': 'LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs, as well as tools to work with prompts, chains, agents, and memory. LangChain is designed to help developers build applications that can reason, plan, and execute tasks using natural language.'})

print(final)

chain.get_graph().print_ascii()