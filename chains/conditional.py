from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text as either "
        "positive or negative.\n\n"
        "Feedback: {feedback}\n\n"
        "{format_instruction}\n"
        "Remember: Only output a valid JSON object."
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()},
)


classifier_chain = prompt | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback: {feedback}.',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback: {feedback}.',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Invalid sentiment")  # fallback
)


chain = classifier_chain | branch_chain

final = chain.invoke({'feedback': 'The product quality is excellent and delivery was prompt!'})
print(final)

chain.get_graph().print_ascii()

