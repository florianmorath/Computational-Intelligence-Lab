from surprise import Reader
import helpers


class CustomReader(Reader):
    """
    Custom reader to read the given data for surprise lib
    """
    
    def __init__(self):
        self.skip_lines = 1
        self.rating_scale = (1, 5)
        # The rating offset (if rating scale lower bound is different than 1)
        self.offset = 0
    
    def parse_line(self, line):
        uid, iid, r = helpers.parse_line(line)
        timestamp = None
        return uid, iid, float(r), timestamp

    
def get_ratings_from_predictions(predictions):
    """
    Generates tuples of the format (row, col, rating) from predictions.
    """
    for pred in predictions:
        yield pred.uid, pred.iid, pred.est