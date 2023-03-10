import uvicorn
from fastapi import FastAPI, Request, status
from models import *
from utils.utils import predict
import torch
import os
import sys


app = FastAPI()



@app.post('/api/predict',response_model=InferenceResponse)
def predict_image(request: Request, body: InferenceInput):
    return predict(body.image)


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=True, 
                )