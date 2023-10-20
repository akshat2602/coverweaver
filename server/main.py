import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(threadName)s] "
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(handlers=[console_handler])
logger = logging.getLogger(__name__)


app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return "pong"


@app.get("/health")
def health():
    return "ok"
