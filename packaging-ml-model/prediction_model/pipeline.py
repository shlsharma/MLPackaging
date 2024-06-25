'''
Using the scikit learn pipeline.
The overall idea of the pipeline is to assemble the various steps
which we can cross validate as when we are working
'''
from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('Mean Imputation', pp.MeanImputer(config.NUM_FEATURES)),
        ('Mode Imputation', pp.ModeImputer(config.CAT_FEATURES)),
        ('Domain Processing', pp.DomainProcessing(config.FEATURE_TO_MODIFY, config.FEATURE_TO_ADD)),
        ('Drop Features', pp.DropColumns(config.DROP_FEATURES)),
        ('LabelEncoding', pp.CustomLabelEncoder(config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)