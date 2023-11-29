# some_params

tk = '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'
num_speaker = 2
use_enh = False

ASR_PORT = ['8001', '8002', '8003']
ASR_URIS = ['http://127.0.0.1:'+x+'/api/asr' for x in ASR_PORT]
CLOVA_URI = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
CLOVA_HEADERS = {"X-NCP-APIGW-API-KEY-ID" : "ymydhcvipo", "X-NCP-APIGW-API-KEY" : "BZFREXShaW0SyoFdWO35UTev6NBPRLM0VoyjIGdE","Content-Type": "application/json"}
GPT_KEY = "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"
GPT_URI = "https://api.openai.com/v1/chat/completions"
GPT_HEADER = headers={"Authorization": f"Bearer {GPT_KEY}"}
UPLOAD_DIRECTORY = "./audio"
TEMP_DIRECTORY = "./tempaudios"