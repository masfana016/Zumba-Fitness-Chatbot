from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint (
     repo_id = "HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
)

langchainPrompt = PromptTemplate.from_template('recipe of delicious {type} cake')

chat_model = ChatHuggingFace (llm=llm)

# prompt  = "chocolate cake recipe"

prompt_output = langchainPrompt.invoke({"type": "chocolate"})

result = chat_model.invoke(prompt_output)

print (result)











# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os

# load_dotenv()

# llm = HuggingFaceEndpoint (
#      repo_id = "HuggingFaceH4/zephyr-7b-beta",
#     huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
# )

# chat_model = ChatHuggingFace(llm=llm)

# langchainPrompt = PromptTemplate.from_template("give me five islamic names of {baby}}")
# # prompt = "give me five islamic names of baby boy"

# prompt_outcome = langchainPrompt.invoke({"baby":"girl"})

# result = chat_model.invoke(prompt_outcome)

# print (result)