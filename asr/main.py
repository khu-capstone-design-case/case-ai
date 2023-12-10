from fastapi import FastAPI, UploadFile, Form, File
from transformers import pipeline
from classes import *
from ..separation.setting import *
import soundfile as sf
import io

pipe = pipeline("automatic-speech-recognition", model="Thebreeze129/case-ai", token=TOKEN)

app = FastAPI()

@app.get("/")
#async def root():
def root():
    return {"message" : "Hello, world!"}


@app.post("/api/asr")
async def asr(seq:int=Form(), user:str=Form(),file:UploadFile=File()):
    contents = await file.read()
    audio_arr, _ = sf.read(io.BytesIO(contents))
    result = pipe(audio_arr)
    message = result["text"]
    return ASR_Result(**{'seq':seq, 'user':user, 'message':message})

