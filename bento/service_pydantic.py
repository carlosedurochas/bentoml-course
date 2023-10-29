import numpy as np
import pandas as pd
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON
from typing import Optional
from pydantic import BaseModel

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner])

class IrisFeature(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float

    # optional field
    request_id: Optional[int]

    # use pydantic's validator to check input data
    class Config:
        extra = "forbid"

input_spec = JSON(pydantic_model=IrisFeature)

@svc.api(input=input_spec, output=NumpyNdarray())
def classify(input_data: IrisFeature) -> np.ndarray:
    if input_data.request_id:
        print(f"request_id: {input_data.request_id}")
    input_df = pd.DataFrame([input_data.model_dump(exclude={"request_id"})])
    return iris_clf_runner.predict.run(input_df)