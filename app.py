from fastapi import FastAPI,Form
from fastapi.staticfiles import StaticFiles
import uvicorn 

from helpers.helper_functions import *
from main.models import ModelClass

model = ModelClass()

api_app = FastAPI(title="api app")
app = FastAPI(title="main app")

app.mount("/", StaticFiles(directory="page", html=True), name="index")

@app.get("/get_likes")
async def get_likes(
    element_content: str = Form(...),
    alt_text: str = Form(...),
    hidden_text: str = Form(...)
):
    # Process the received data
    # You can store it in a database, analyze it, etc.
    # For demonstration, we'll just return the received data
    data = {
        "Element Content": element_content,
        "Alt Text": alt_text,
        "Hidden Text": hidden_text
    }
    
    return data

@app.post("/perform_ml_prediction")
async def perform_ml_prediction():
    # Call the /get_likes endpoint to retrieve data
    response = await get_likes()
    data = response.json()
    
    # Perform your machine learning prediction using the retrieved data
    prediction_result = model.create_embeddings(data)
    
    # Combine the original data with the prediction result
    #ml_result = model.perform_clustering(prediction_result)
    
    return prediction_result


if __name__ == "__main__":
    uvicorn.run(app,port=8080,host='0.0.0.0')