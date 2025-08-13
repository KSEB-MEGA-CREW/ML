# extract mediapipe keypoints
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, List
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class KeypointExtractor:
    def __init__(self):
        # Mediapipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.MAX_HANDS,
            min_detection_confidence=settings.HAND_CONFIDENCE,
            min_tracking_confidence=0.5,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=settings.POSE_CONFIDENCE,
            min_tracking_confidence=0.5,
        )

    def extract_keypoints(self, image: np.ndarray) -> Dict[str, any]:
        """이미지에서 키포인트 추출"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 손 키포인트 추출
        hand_results = self.hands.process(rgb_image)
        hand_landmarks = self._process_hand_landmarks(hand_results)

        # 포즈 키포인트 추출
        pose_results = self.pose.process(rgb_image)
        pose_landmarks = self._process_pose_landmarks(pose_results)

        return {"hand_landmarks": hand_landmarks, "pose_landmarks": pose_landmarks}

    def _process_hand_landmarks(self, results) -> Dict[str, Optional[List]]:
        """손 랜드마크 처리"""
        hand_landmarks = {"left_hand": None, "right_hand": None}

        if results.multi_hand_landmarks:
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                # 손 구분
                handedness = results.multi_handedness[idx].classification[0].label

                # 랜드마크 좌표 추출
                landmark_list = []
                for landmark in landmarks.landmark:
                    landmark_list.append([landmark.x, landmark.y, landmark.z])

                if handedness == "Left":
                    hand_landmarks["left_hand"] = landmark_list
                else:
                    hand_landmarks["right_hand"] = landmark_list

        return hand_landmarks

    def _process_pose_landmarks(self, results) -> Optional[List]:
        """포즈 랜드마크 처리 (상체만)"""
        if not results.pose_landmarks:
            return None

        # 상체 키포인트 인덱스 (어깨, 팔꿈치, 손목, 얼굴 일부)
        upper_body_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,  # 얼굴
            11,
            12,
            13,
            14,
            15,
            16,  # 상체
        ]

        pose_landmarks = []
        for idx in upper_body_indices:
            landmark = results.pose_landmarks.landmark[idx]
            pose_landmarks.append(
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
            )

        return pose_landmarks
