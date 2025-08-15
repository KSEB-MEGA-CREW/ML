import cv2
import mediapipe as mp
import numpy as np
import asyncio
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class KeypointExtractor:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=settings.PROCESS_POOL_SIZE)

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

    async def extract_keypoints(self, image: np.ndarray) -> Dict[str, any]:
        """Extract keypoints from image asynchronously"""
        try:
            # Convert to RGB asynchronously
            loop = asyncio.get_event_loop()
            rgb_image = await loop.run_in_executor(
                self.executor, cv2.cvtColor, image, cv2.COLOR_BGR2RGB
            )

            # Process hands and pose in parallel
            hand_task = loop.run_in_executor(
                self.executor, self._process_hands, rgb_image
            )
            pose_task = loop.run_in_executor(
                self.executor, self._process_pose, rgb_image
            )

            hand_landmarks, pose_landmarks = await asyncio.gather(hand_task, pose_task)

            return {"hand_landmarks": hand_landmarks, "pose_landmarks": pose_landmarks}

        except Exception as e:
            logger.error(f"Keypoint extraction failed: {e}")
            return {
                "hand_landmarks": {"left_hand": None, "right_hand": None},
                "pose_landmarks": None,
            }

    def _process_hands(self, rgb_image: np.ndarray) -> Dict[str, Optional[List]]:
        """Process hand landmarks"""
        try:
            hand_results = self.hands.process(rgb_image)
            return self._process_hand_landmarks(hand_results)
        except Exception as e:
            logger.error(f"Hand processing failed: {e}")
            return {"left_hand": None, "right_hand": None}

    def _process_pose(self, rgb_image: np.ndarray) -> Optional[List]:
        """Process pose landmarks"""
        try:
            pose_results = self.pose.process(rgb_image)
            return self._process_pose_landmarks(pose_results)
        except Exception as e:
            logger.error(f"Pose processing failed: {e}")
            return None

    def _process_hand_landmarks(self, results) -> Dict[str, Optional[List]]:
        """Process hand landmarks"""
        hand_landmarks = {"left_hand": None, "right_hand": None}

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                if idx < len(results.multi_handedness):
                    # Determine hand type
                    handedness = results.multi_handedness[idx].classification[0].label

                    # Extract landmark coordinates
                    landmark_list = []
                    for landmark in landmarks.landmark:
                        landmark_list.append([landmark.x, landmark.y, landmark.z])

                    if handedness == "Left":
                        hand_landmarks["left_hand"] = landmark_list
                    else:
                        hand_landmarks["right_hand"] = landmark_list

        return hand_landmarks

    def _process_pose_landmarks(self, results) -> Optional[List]:
        """Process pose landmarks (upper body only)"""
        if not results.pose_landmarks:
            return None

        # Upper body keypoint indices
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
            10,  # Face
            11,
            12,
            13,
            14,
            15,
            16,  # Upper body
        ]

        pose_landmarks = []
        for idx in upper_body_indices:
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                pose_landmarks.append(
                    [landmark.x, landmark.y, landmark.z, landmark.visibility]
                )

        return pose_landmarks

    def __del__(self):
        """Clean up resources"""
        try:
            if hasattr(self, "hands"):
                self.hands.close()
            if hasattr(self, "pose"):
                self.pose.close()
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
