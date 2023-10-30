import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader


st.title("CoverWeaver")

with st.sidebar:
    st.subheader("OpenAI API Key")
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


def generate_summary(blogs, llm, company_name="Cloudflare"):
    map_template = """Extract technical information from the articles below 
    to help me write a cover letter when applying to software developer 
    roles at Cloudflare.
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


def generate_response(input_blogs, jd_url, file):
    progress_text = "Parsing Resume. Please wait."
    my_bar = st.progress(0, text=progress_text)
    resume = parse_resume(file)
    progress_text = "Getting blogs data. Please wait."
    my_bar.progress(5, text=progress_text)
    blogs = blog_loader(input_blogs)
    progress_text = "Getting job description data. Please wait."
    my_bar.progress(20, text=progress_text)
    jd = jd_loader(jd_url)
    progress_text = "Generating blog summaries. Please wait."
    my_bar.progress(30, text=progress_text)
    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key
    )
    summaryDocs = generate_summary(blogs=blogs, llm=llm)
    progress_text = "Generating cover letter. Please wait."
    my_bar.progress(80, text=progress_text)
    fin_template = """Here is a summary of blogs from cloudflare :
    {summaryDocs}
    Summary of documents ends here.
    Here is my resume:
    {resume}
    Here is the Job description:
    {jd}
    Generate a great cover letter for me to apply to Cloudflare only using the above inputs,
    demonstrating excitement about work shown in the blogs. ONLY mention the work I've done in my resume, do not make things up.
    Link my demonstrated experience in my resume to technical challenges faced at Cloudflare for the topics in the job description.
    Balance the cover letter so that the content from the blogs and my resume is used equally, don't let one input dominate the other.
    Being truthful is of the highest importance. Do not say I have done something not present in my resume.
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
    blog_1_url = st.text_input("Blog Link", "Enter blog link:", key="blog_1")
    blog_2_url = st.text_input("Blog Link", "Enter blog link:", key="blog_2")
    blog_3_url = st.text_input("Blog Link", "Enter blog link:", key="blog_3")
    jd_url = st.text_input("Job Link", "Enter job link:", key="jd")
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
        resp = generate_response([blog_1_url, blog_2_url, blog_3_url], jd_url, file)

if resp:
    st.write(resp)
