# ai_models/face_recognition/base_model.py

from abc import ABC, abstractmethod
import numpy as np

class FaceRecognitionModel(ABC):
    """
    Abstract base class for face recognition models.
    """

    @abstractmethod
    def load_model(self):
        """
        Load the model and any dependencies.
        """
        pass

    @abstractmethod
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Given a face image, return the embedding vector.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name.
        """
        pass