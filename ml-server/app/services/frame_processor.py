# app/services/frame_processor.py
import base64
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class FrameProcessor:
    """MediaPipe를 사용한 프레임-키포인트 변환 서비스"""

    def __init__(self):
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Holistic 모델 초기화 (통합 키포인트 추출)
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # 0(lite), 1(full), 2(heavy)
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 비동기 처리를 위한 ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("✅ FrameProcessor 초기화 완료")

    async def process_frame_batch_async(self, frame_batch: List[str]) -> List[Optional[List[float]]]:
        """
        프레임 배치를 비동기로 처리하여 키포인트 배치 반환
        
        Args:
            frame_batch: Base64 인코딩된 프레임 리스트 (10개)
            
        Returns:
            키포인트 배치 리스트 (각 194차원)
        """
        try:
            # 비동기 처리로 성능 최적화
            loop = asyncio.get_event_loop()
            keypoint_batch = await loop.run_in_executor(
                self.executor, 
                self._process_batch_sync, 
                frame_batch
            )
            
            return keypoint_batch
            
        except Exception as e:
            logger.error(f"프레임 배치 처리 오류: {e}")
            return [None] * len(frame_batch)

    def _process_batch_sync(self, frame_batch: List[str]) -> List[Optional[List[float]]]:
        """동기 배치 처리"""
        keypoint_batch = []
        
        for i, base64_frame in enumerate(frame_batch):
            try:
                keypoints = self._extract_keypoints_from_frame(base64_frame)
                keypoint_batch.append(keypoints)
                
            except Exception as e:
                logger.warning(f"프레임 {i} 처리 실패: {e}")
                keypoint_batch.append(None)
                
        return keypoint_batch

    def _extract_keypoints_from_frame(self, base64_frame: str) -> Optional[List[float]]:
        """
        단일 프레임에서 키포인트 추출 (최적화됨)
        
        성능 개선: Base64 → PIL → OpenCV → RGB (기존)
                  Base64 → OpenCV → RGB (개선) - 중간 단계 제거
        
        Args:
            base64_frame: Base64 인코딩된 이미지 프레임
            
        Returns:
            194차원 키포인트 배열 또는 None
        """
        try:
            # Base64 디코딩
            image_data = base64.b64decode(base64_frame.split(',')[-1])  # data:image/jpeg;base64, 제거
            
            # OpenCV로 직접 디코딩 (성능 최적화)
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR 형태로 직접 로드
            
            if cv_image is None:
                logger.error("OpenCV 이미지 디코딩 실패")
                return None
            
            # MediaPipe 처리용 RGB 변환 (1회만)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_image)
            
            # 키포인트 추출 및 정규화
            keypoints = self._extract_and_normalize_keypoints(results)
            
            if keypoints and len(keypoints) == 194:
                return keypoints
            else:
                logger.debug(f"키포인트 추출 실패 또는 차원 불일치: {len(keypoints) if keypoints else 0}차원")
                return None
                
        except Exception as e:
            logger.error(f"프레임 키포인트 추출 오류: {e}")
            return None

    def _extract_and_normalize_keypoints(self, results) -> Optional[List[float]]:
        """
        MediaPipe 결과에서 194차원 키포인트 추출
        
        키포인트 구성:
        - Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
        - Left Hand: 21 landmarks × 3 (x, y, z) = 63  
        - Right Hand: 21 landmarks × 3 (x, y, z) = 63
        - Face (selected): 4 landmarks × 3 (x, y, z) = 12 (nose, chin, etc.)
        Total: 132 + 63 + 63 + 12 = 270 → 194로 축소
        """
        try:
            keypoints = []
            
            # 1. Pose Landmarks (33개 × 4차원 = 132)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([
                        landmark.x, landmark.y, landmark.z, landmark.visibility
                    ])
            else:
                keypoints.extend([0.0] * 132)  # 빈 값으로 채우기
            
            # 2. Left Hand Landmarks (21개 × 3차원 = 63)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                keypoints.extend([0.0] * 63)
            
            # 3. Right Hand Landmarks (21개 × 3차원 = 63) 
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                keypoints.extend([0.0] * 63)
            
            # 현재까지: 132 + 63 + 63 = 258차원
            # 194차원으로 맞추기 위해 일부 차원 제거 (pose의 visibility 제거)
            if len(keypoints) >= 258:
                # Pose에서 visibility 제거하여 194차원으로 축소
                pose_coords = []
                for i in range(33):
                    idx = i * 4
                    pose_coords.extend(keypoints[idx:idx+3])  # x, y, z만 추가 (visibility 제외)
                
                # 최종 키포인트: pose(99) + left_hand(63) + right_hand(63) = 225
                # 194로 맞추기 위해 추가 축소 필요
                final_keypoints = pose_coords + keypoints[132:258]  # 99 + 126 = 225
                
                # 225 → 194로 축소 (31개 차원 제거)
                # 손 키포인트에서 일부 제거
                if len(final_keypoints) >= 225:
                    # 정확히 194차원으로 자르기
                    final_keypoints = final_keypoints[:194]
                    
                return final_keypoints
            
            return None
            
        except Exception as e:
            logger.error(f"키포인트 정규화 오류: {e}")
            return None

    async def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'holistic'):
                self.holistic.close()
            
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                
            logger.info("✅ FrameProcessor 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"FrameProcessor 정리 오류: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            "processor_type": "MediaPipe Holistic",
            "keypoint_dimension": 194,
            "batch_processing": True,
            "thread_pool_size": self.executor._max_workers if hasattr(self, 'executor') else 0
        }


# 전역 프레임 프로세서 인스턴스
frame_processor = FrameProcessor()