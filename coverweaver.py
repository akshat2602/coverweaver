import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_db():
    # Initialize the database
    global db
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(embedding_function=embeddings)


st.title("CoverWeaver")
st.link_button("GitHub", "https://github.com/akshat2602/coverweaver/tree/streamlit")

st.text("Generate a cover letter for a job application using OpenAI's GPT-3")

with st.sidebar:
    st.subheader("OpenAI API Key")
    global openai_api_key
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


def parse_resume(file):
    resume = ""
    if file is not None:
        # Write the file to a temp location
        with open("temp_resume.pdf", "wb") as f:
            f.write(file.read())
        resumeloader = PyPDFLoader("temp_resume.pdf")
        resume = resumeloader.load()
    return resume


def generate_summary(blogs, llm, company_name):
    map_template = """Extract technical information from the articles below
    to help me write a cover letter when applying to software developer
    roles at {company_name}.
    "{blogs}"
    SUMMARY:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    # Reduce
    reduce_template = """The following is set of summaries from blog articles:
    {doc_summaries}
    Take these and distill it into a technical summary :
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    # Takes a list of documents, combines them into a single string, and
    # passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="blogs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    summaryDocs = map_reduce_chain.run(blogs)
    return summaryDocs


def jd_loader(url):
    jd_loader = WebBaseLoader(url)
    return jd_loader.load()


def blog_loader(urls=[]):
    b_loader = WebBaseLoader(urls)
    return b_loader.load()


def add_docs_to_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    global db
    db.add_documents(documents)
    return


def get_question_prompt(company_name, resume):
    from langchain.chains import LLMChain

    prompt_template = """Here is a resume . The resume ends when you see the phrase 
    'resume ends here.':
    {resume}
    Resume ends here.
    You are an AI language model assistant. The user has the above resume.
    The user also has a vector database with some data scraped from the
    {companyName} blog.
    Generate five queries to use in similarity search to retrieve content most relevant
    and matching to the user's provided resume.
    By generating multiple perspectives on the user question, your goal is to help the
    user overcome some of the limitations
    of the distance-based similarity search. Provide these alternative queries
    separated by newlines.
    Make sure that the queries are not too similar to each other, 
    our goal is to fetch different unique results using these queries.
    don't do anything that is not mentioned in the prompt."""
    llm = ChatOpenAI(
        temperature=0, model="gpt-4-1106-preview", openai_api_key=openai_api_key
    )
    prompt_object = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt_object)
    questions = llm_chain.run(
        {"resume": resume, "companyName": company_name}
    )
    print(questions)
    refined_prompt = """
    {questions}
    From the above text, just give me a newline separated list of the 
    queries to be used in similarity search,
    remove terms like 'articles, retrieve, fetch, content, find'
    """

    refined_prompt_object = PromptTemplate.from_template(refined_prompt)

    refined_llm_chain = LLMChain(llm=llm, prompt=refined_prompt_object)
    refined_questions = refined_llm_chain.run(
        {"questions": questions}
    )
    print(refined_questions)
    refined_questions = refined_questions.split("\n")

    sources_to_summarize = []
    for refined_refined_question in refined_questions:
        retriever = db.as_retriever(llm=llm, search_type="mmr")
        unique_doc = retriever.get_relevant_documents(query=refined_refined_question)
        sources_to_summarize.append(unique_doc[0].metadata["source"])

    print(sources_to_summarize)
    return sources_to_summarize


def extract_blogs_using_company_name(company_name, resume):
    # Read json file
    with open("blogs.json", "r") as myfile:
        data = myfile.read()
    # parse file
    obj = json.loads(data)
    # Extract blogs for the company_name
    langchain_docs_list = []
    for blog in obj[company_name]:
        doc = Document(page_content=blog["text"], metadata={"source": blog["url"]})
        langchain_docs_list.append(doc)

    add_docs_to_db(langchain_docs_list)
    blog_sources_list = get_question_prompt(company_name, resume)

    return blog_sources_list

def generate_response(company_name, jd_url, file):
    progress_text = "Parsing Resume. Please wait."
    my_bar = st.progress(0, text=progress_text)
    resume = parse_resume(file)
    progress_text = "Getting blogs data. Please wait."
    my_bar.progress(5, text=progress_text)
    input_blogs = extract_blogs_using_company_name(company_name, resume)
    blogs = blog_loader(input_blogs)
    progress_text = "Getting job description data. Please wait."
    my_bar.progress(20, text=progress_text)
    jd = jd_loader(jd_url)
    progress_text = "Generating blog summaries. Please wait."
    my_bar.progress(30, text=progress_text)
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-4-1106-preview", openai_api_key=openai_api_key
    )
    summaryDocs = generate_summary(blogs=blogs, llm=llm, company_name=company_name)
    progress_text = "Generating cover letter. Please wait."
    my_bar.progress(80, text=progress_text)
    fin_template = """Here is a summary of blogs from {company_name}} :
    {summaryDocs}
    Summary of documents ends here.
    Here is my resume:
    {resume}
    Here is the Job description:
    {jd}
    Generate a great cover letter for me to apply to {company_name} only using the
    above inputs,demonstrating excitement about work shown in the blogs.
    ONLY mention the work I've done in my resume, do not make things up.
    Link my demonstrated experience in my resume to technical challenges faced
    at {company_name} for the topics in the job description.
    Balance the cover letter so that the content from the blogs and my resume is
    used equally, don't let one input dominate the other.
    Being truthful is of the highest importance. Do not say I have done
    something not present in my resume.
    Make it sound more human but still professional.
    Cover letter:"""
    fin_prompt = PromptTemplate.from_template(fin_template)
    fin_chain = LLMChain(llm=llm, prompt=fin_prompt)
    coverletter = fin_chain.run(
        {"summaryDocs": summaryDocs, "resume": resume, "jd": jd}
    )
    progress_text = "Cover Letter Generated. Please wait."
    my_bar.progress(100, text=progress_text)
    my_bar.empty()
    st.info("Cover Letter Generated!")

    return coverletter


resp = None  # Initialize to None

st.subheader("Inputs")
with st.form("my_form"):
    company_name = st.text_input("Company Name", key="company_name")
    jd_url = st.text_input("Job Link", key="jd")
    file = st.file_uploader(
        "Resume",
        accept_multiple_files=False,
        on_change=None,
        label_visibility="visible",
        type=["pdf"],
        key="resume",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        init_db()
        resp = generate_response(company_name, jd_url, file)

if resp:
    st.write(resp)
