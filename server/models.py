from pydantic import BaseModel
from fastapi import UploadFile


class LetterReq(BaseModel):
    jobDescription: str
    blogLinks: list[str]
    resume: UploadFile


class LetterResp(BaseModel):
    coverLetter: str
