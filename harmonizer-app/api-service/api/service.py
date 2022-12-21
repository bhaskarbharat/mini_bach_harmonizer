from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
from api import tracker
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model


# Setup FastAPI app
app = FastAPI(
    title="API Server",
    description="API Server",
    version="v1"
)

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # Startup tasks
    # Start the tracker service
    # asyncio.create_task(tracker.download_model_weights())
    # tracker.download_model_weights()
    pass

# Routes
@app.get("/")
async def get_index():
    return {
        "message": "Welcome to the API Service"
    }


@app.post("/predict")
async def predict(
        sequence: dict = {}
):
    # print("predict file:", len(file), type(file))
    print("predict file:", sequence, type(sequence))

    # # Save the input.json
    # with TemporaryDirectory() as input_dir:
    #     input_path = os.path.join(input_dir, "input.json")
    #     with open(input_path, "wb") as output:
    #         output.write(file)

    # Make prediction
    prediction_results = model.make_prediction(sequence)

    return prediction_results
    