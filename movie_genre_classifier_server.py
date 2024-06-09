# FastAPI app
import json
import os
from typing import List

import joblib
from argparse import ArgumentParser
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from pydantic import BaseModel
from starlette.responses import JSONResponse

from movie_genre_classifier_predict import predict_movie_genres

app = FastAPI()


# Pydantic model for request body
class MovieData(BaseModel):
    # title: str
    plot_summary: str
    # languages: str
    # genres: str
    # countries: str


@app.post("/predict-genres")
async def predict_movies_genres(inputs: List[MovieData]):
    inputs = [i.dict() for i in inputs]
    # Get predictions
    predictions = predict_movie_genres(xgb_model, data_pipe, inputs, y_cols)
    # Convert numpy float32 to Python float
    predictions = {k: v.item() for k, v in predictions.items()}
    return JSONResponse(content=predictions)


@app.get("/ping")
async def ping():
    print('received ping')
    return 'pong'


if __name__ in ('__main__', os.path.basename(__file__).split('.')[0]):
    parser = ArgumentParser()
    parser.add_argument('data_pipe_path', type=str, help='Path to the preprocessing pipeline')
    parser.add_argument('model_path', type=str, help='Path to the trained XGBoost model')
    parser.add_argument('--port', type=int, default=8001, help='Port for the server')
    args = parser.parse_args()

    # Load the preprocessing pipeline and the trained XGBoost model
    _, data_pipe = joblib.load(args.data_pipe_path)
    data_pipe_ext = os.path.basename(args.data_pipe_path).rsplit('.', 1)[1]
    y_cols_path = args.data_pipe_path.replace(f'.{data_pipe_ext}', '_y_cols.json')
    with open(y_cols_path, 'r') as f:
        y_cols = json.load(f)
    xgb_model = joblib.load(args.model_path)

if __name__ == '__main__':
    uvicorn.run("movie_genre_classifier_server:app", host='0.0.0.0', port=args.port)
