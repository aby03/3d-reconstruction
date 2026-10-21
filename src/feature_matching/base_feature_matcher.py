from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureMatcher(ABC):
    """
    Abstract base class for feature extractors and matchers
    """
    @abstractmethod
    def match(self, img1: np.ndarray, img2: np.ndarray):
        """
        Matches features between two images.
        
        Args:
            img1: First image (H x W x C, float32 or uint8)
            img2: Second image (H x W x C, float32 or uint8)
            
        Returns:
            pts1: Matched points in image 1 (N x 2 numpy array of [u, v])
            pts2: Matched points in image 2 (N x 2 numpy array of [u, v])
            matches: OpenCV matches corresponding to pts1 and pts2
            scores: Descriptor distances for each match (N,)
            keypoints1: Detected OpenCV keypoints in image 1
            keypoints2: Detected OpenCV keypoints in image 2
        """
        pass
