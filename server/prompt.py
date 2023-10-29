from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

map_template = """Extract technical information from the articles
below to help me write a cover letter when applying to software
developer roles at Cloudflare :
"{docs}"
SUMMARY:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)
