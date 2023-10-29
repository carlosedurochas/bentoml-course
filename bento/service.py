import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
my_python_runner = bentoml.picklable_model.get("my_python_model:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner, my_python_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def square(input_series: np.ndarray) -> np.ndarray:
    result = my_python_runner.run(input_series)
    return result