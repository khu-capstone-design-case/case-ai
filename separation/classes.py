from pydantic import BaseModel
import numpy as np
from typing import List

class VoiceFile:
    def __init__(self, user, speakerNum, filepath):
        self.user = user
        self.speakerNum = speakerNum
        self.filepath = filepath

class Message(BaseModel):
    seq : int
    speaker : str
    message : str = None
    startTime : float
    endTime : float
    positive : float = None
    negative : float = None
    neutral : float = None
    audio : np.ndarray = None
    mix : bool

    class Config:
        arbitrary_types_allowed = True

class res_Content(BaseModel):
    seq : int
    speaker : str
    message : str = None
    startTime : float
    endTime : float
    positive : float = None
    negative : float = None
    neutral : float = None

class AudioResponse(BaseModel):
    fileName : str
    user : str
    speakerNum : int
    length : float
    positive : float
    negative : float
    neutral : float
    summary : str
    message : List[res_Content]

class Script(BaseModel):
    script : List[str]