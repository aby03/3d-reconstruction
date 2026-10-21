import cv2
import numpy as np

from src.feature_matching.base_feature_matcher import BaseFeatureMatcher

class SIFTMatcher(BaseFeatureMatcher):
    def __init__(self, ratio_threshold: float = 0.75):
        self.sift = cv2.SIFT_create()
        self.ratio_threshold = ratio_threshold
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim not in (2, 3):
            raise ValueError(f"Image dimensions {image.shape} not supported")

        if image.dtype != np.uint8:
            image = np.nan_to_num(image)
            if image.size and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return image
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        raise ValueError(f"Image channel count {image.shape[2]} not supported")

    def match(self, img1: np.ndarray, img2: np.ndarray):
        gray1 = self._to_grayscale_uint8(img1)
        gray2 = self._to_grayscale_uint8(img2)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            empty_points = np.empty((0, 2), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            return empty_points, empty_points.copy(), [], empty_scores, kp1, kp2

        knn_matches = self.bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for candidates in knn_matches:
            if len(candidates) < 2:
                continue
            m, n = candidates
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        scores = np.float32([m.distance for m in good_matches])

        return pts1, pts2, good_matches, scores, kp1, kp2