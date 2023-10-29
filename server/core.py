from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain


loader = WebBaseLoader(
    [
        "https://blog.cloudflare.com/unbounded-memory-usage-by-tcp-for-receive-buffers-and-how-we-fixed-it/",
        "https://blog.cloudflare.com/building-cloudflare-on-cloudflare/",
        "https://blog.cloudflare.com/how-cloudflare-runs-prometheus-at-scale/",
    ]
)
resumeloader = PyPDFLoader("./OmkarPradeep_Rajwade_resume.pdf")
jdLoader = WebBaseLoader(
    "https://boards.greenhouse.io/cloudflare/jobs/5366615?gh_jid=5366615"
)
resume = resumeloader.load()
docs = loader.load()
jd = jdLoader.load()
