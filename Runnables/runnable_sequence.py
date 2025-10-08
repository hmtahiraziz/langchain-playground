from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate a joke about {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template='Explain this joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)

chain.invoke({'topic': 'programming'})