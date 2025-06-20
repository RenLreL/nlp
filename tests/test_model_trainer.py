import unittest
import numpy as np
from collections import namedtuple
from sklearn.metrics import accuracy_score, f1_score
from src.training.model_trainer import ModelTrainer

PredictionOutput = namedtuple('PredictionOutput', ['predictions', 'label_ids'])

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        """Provides a ModelTrainer instance with a dummy config."""
        dummy_config = {'key': 'value'}
        self.model_trainer = ModelTrainer(config=dummy_config)

    def test_compute_metrics(self):
        """
        Tests the _compute_metrics function for correctness.
        """
        logits = np.array([
            [3.1, 0.1, 0.1],  # Prediction: 0
            [0.1, 4.5, 0.1],  # Prediction: 1
            [0.1, 0.2, 5.3],  # Prediction: 2
            [2.1, 1.1, 0.1],  # Prediction: 0
        ])
        labels = np.array([0, 1, 1, 0])  # True labels. Note: the 3rd prediction is wrong
        preds = np.argmax(logits, axis=-1)  # Expected predictions: [0, 1, 2, 0]

        mock_pred = PredictionOutput(predictions=logits, label_ids=labels)
        metrics = self.model_trainer._compute_metrics(mock_pred)
        self.assertIsInstance(metrics, dict)
        expected_keys = ['accuracy', 'f1', 'precision', 'recall']
        for key in expected_keys:
            self.assertIn(key, metrics)

        expected_acc = accuracy_score(labels, preds)
        expected_f1 = f1_score(labels, preds, average='weighted')

        self.assertAlmostEqual(metrics['accuracy'], expected_acc)
        self.assertAlmostEqual(metrics['f1'], expected_f1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)