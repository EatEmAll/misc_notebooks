# FastAPI app
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from movie_genre_classifier_predict import predict_movie_genres

app = FastAPI()


# Pydantic model for request body
class MovieData(BaseModel):
    title: str
    plot_summary: str
    languages: str
    genres: str
    countries: str
    # Add other necessary fields with appropriate types


@app.post("/predict-genres/")
def predict_movies_genres(movie_datas: List[MovieData]):
    # Convert the incoming MovieData to a list of dicts
    movie_data_dict = [movie_data.dict() for movie_data in movie_datas]
    try:
        # Get predictions
        predictions = predict_movie_genres(movie_data_dict)
        return predictions

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# To run the server, use the following command in the terminal:
if __name__ == '__main__':
    from argparse import ArgumentParser
    import uvicorn

    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port for the server')
    args = parser.parse_args()

    uvicorn.run("movie_genre_classifier_server:app", host='0.0.0.0', port=args.port)
