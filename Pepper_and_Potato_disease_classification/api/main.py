from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2

import tensorflow as tf
MODEL=tf.keras.models.load_model("./models/1")
class_names=["Pepper Bell Bacterial Spot","Pepper Bell Healthy","Potato Early Blight", "Potato Late Blight", "Potato Healthy"]
app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "hello"


def read_file_as_image(data):
    img=np.array(Image.open(BytesIO(data)))
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image=read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)

    prediction=MODEL.predict(image_batch)
    print(np.argmax(prediction[0]))
    predicted_class=class_names[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])



    
    return {"class":predicted_class,
                "confidence":float(confidence)}  

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)