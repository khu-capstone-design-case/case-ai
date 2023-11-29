from fastapi import FastAPI, UploadFile, Form, File
from transformers import pipeline
from classes import *

pipe = pipeline("automatic-speech-recognition", model="SungBeom/whisper-small-ko")

app = FastAPI()

@app.get("/")
#async def root():
def root():
    return {"message" : "Hello, world!"}


@app.post("/api/asr")
async def asr(seq:int=Form(), user:str=Form(),file:UploadFile=File()):
    contents = await file.read()
    tempfilename = "./tempaudios"+user+"_temp"+str(seq)+".wav"
    with open(tempfilename, "wb") as fp:
        fp.write(contents)
    result = pipe(tempfilename)
    message = result["text"]
    return ASR_Result(**{'seq':seq, 'user':user, 'message':"당신은 나의 사랑입니다. 하지만 그만하려합니다."})

