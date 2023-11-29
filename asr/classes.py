from pydantic import BaseModel

class ASR_Result(BaseModel):
    seq : int
    message : str
    user : str