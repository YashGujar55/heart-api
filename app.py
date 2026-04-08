from fastapi import FastAPI, HTTPException, Header

from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
import logging
API_KEY = "12345"
# Create app
app = FastAPI()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trtbps: int
    chol: int
    fbs: int
    restecg: int
    thalachh: int
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int

@app.get("/")
def home():
    return {"message": "Heart Disease API"}

# 🔥 UPDATED PREDICT FUNCTION
@app.post("/predict")
def predict(data: HeartData, x_api_key: str = Header(...)):
    try:
        # 🔐 Check API key
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        logging.info(f"Incoming data: {data}")

        input_data = np.array([[ 
            data.age, data.sex, data.cp, data.trtbps,
            data.chol, data.fbs, data.restecg, data.thalachh,
            data.exng, data.oldpeak, data.slp, data.caa, data.thall
        ]])

        prediction = model.predict(input_data)

        logging.info(f"Prediction: {prediction[0]}")

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")