import numpy as np
import pytest

from mlalgorithms.modules.svm import SVM

from sklearn import datasets

from mlalgorithms.utils.dist_fun import visualize_svm

@pytest.mark.differential
def test_model_diff():
    """We wont to compere two different models or any other
        important feature betwen current and prvius version of model
        befor we push changes to repo.
    """
    print("Dummy example")