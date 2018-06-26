
import helpers
import numpy as np
import torch
from spotlight.evaluation import rmse_score
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel


from surprise import AlgoBase
from surprise.model_selection import cross_validate
from surprise_helpers import CustomReader, get_ratings_from_predictions
from surprise import Reader, Dataset


class ExplicitFactorizationModelBase(AlgoBase):
   
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # spotlight model
        user_ids = []
        movie_ids = []
        ratings = []

        for u, i, r in trainset.all_ratings():
            user_ids.append(u)
            movie_ids.append(i)
            ratings.append(r)


        explicit_interactions = Interactions(np.asarray(user_ids, dtype=np.int32), np.asarray(movie_ids, dtype=np.int32), np.asarray(ratings, dtype=np.float32))
        self.model = ExplicitFactorizationModel(loss='regression', embedding_dim=32, n_iter=10, batch_size=256, learning_rate=0.01, l2=0.0, sparse=False)
        self.model.fit(explicit_interactions, verbose=True)

        return self

    def estimate(self, u, i):

        est = self.model.predict(u, np.array(i))[0]

        return est


reader = CustomReader()
filepath = helpers.get_train_file_path()
data = Dataset.load_from_file(filepath, reader=reader)

algo = ExplicitFactorizationModelBase()

cross_validate(algo, data, verbose=True)