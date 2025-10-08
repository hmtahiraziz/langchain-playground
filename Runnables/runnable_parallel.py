from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Generate a tweet about {topic}.',
    input_variables=['topic'],
)

prompt1 =PromptTemplate(
    template='Generate a LinkedIn post about {topic}.',
    input_variables=['topic'],
)

parser = StrOutputParser()

runnable = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt, model, parser),
        'linkedin_post': RunnableSequence(prompt1, model, parser),
    }
)

result = runnable.invoke([{'topic': 'AI'}, {'topic': 'Machine Learning'}])

print(result)