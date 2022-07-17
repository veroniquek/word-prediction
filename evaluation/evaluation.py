
class ModelEvaluator:
    def __init__(self, test_dataset, model, test_split=0.2):
        self.test_split = test_split

        self.test_dataset = test_dataset

        self.model = model

        # train the model
        self.model.train()

    def evaluate_saved_keystrokes(self):
        pass
