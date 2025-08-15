# models/predictor.py
import numpy as np
import asyncio
from typing import Dict, Any
from ..preprocessing.image_processor import ImageProcessor
from ..preprocessing.key_point_extractor import KeypointExtractor
from ..preprocessing.feature_processor import FeatureProcessor
from ..utils.label_loader import LabelLoader
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class SignLanguagePredictor:
    def __init__(self, model):
        self.model = model
        self.image_processor = ImageProcessor()
        self.keypoint_extractor = KeypointExtractor()
        self.feature_processor = FeatureProcessor()
        self.label_loader = LabelLoader()
        self.class_names = None
        self._model_validated = False

    async def _ensure_labels_loaded(self):
        """Load labels if not loaded and validate model compatibility"""
        if self.class_names is None:
            self.class_names = await self.label_loader.load_class_labels()

            # Validate model-label compatibility once
            if not self._model_validated:
                is_compatible = await self.label_loader.validate_model_compatibility(
                    settings.MODEL_NUM_CLASSES
                )
                if not is_compatible:
                    logger.warning("Model and label count mismatch")
                self._model_validated = True

    async def predict(
        self, base64_image: str, session_id: str, frame_index: int
    ) -> Dict[str, Any]:
        """Perform sign language prediction"""
        try:
            # Load and validate labels
            await self._ensure_labels_loaded()

            # 1. Convert Base64 to OpenCV image
            image = self.image_processor.decode_base64(base64_image)
            if image is None or not self.image_processor.validate_image(image):
                return self._create_no_detection_response(session_id, frame_index)

            # 2. Extract keypoints asynchronously
            keypoints = await self.keypoint_extractor.extract_keypoints(image)

            # 3. Create feature vector
            feature_vector = self.feature_processor.create_feature_vector(keypoints)

            # 4. Validate feature vector
            if not self.feature_processor.validate_features(feature_vector):
                return self._create_no_detection_response(session_id, frame_index)

            # 5. Run model inference
            prediction_result = await self._run_inference(feature_vector)

            # 6. Post-process results
            final_result = self._postprocess_prediction(
                prediction_result, keypoints, session_id, frame_index
            )

            return final_result

        except Exception as e:
            logger.error(f"Prediction error: session={session_id}, error={str(e)}")
            return self._create_error_response(session_id, frame_index, str(e))

    async def _run_inference(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Run model inference using TensorFlow"""
        try:
            # Prepare input data
            input_data = np.expand_dims(feature_vector, axis=0)

            # Run prediction asynchronously
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, self.model.predict, input_data
            )

            # Apply softmax if needed
            probabilities = predictions[0]
            if not np.allclose(np.sum(probabilities), 1.0, rtol=1e-5):
                probabilities = self._softmax(probabilities)

            # Calculate top classes
            num_classes = len(self.class_names)
            top_k = min(3, num_classes)
            top_indices = np.argsort(probabilities)[-top_k:][::-1]

            return {
                "probabilities": probabilities,
                "top_classes": top_indices,
                "top_scores": probabilities[top_indices],
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def _softmax(self, x):
        """Apply softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _postprocess_prediction(
        self,
        prediction_result: Dict,
        keypoints: Dict,
        session_id: str,
        frame_index: int,
    ) -> Dict[str, Any]:
        """Post-process prediction results"""
        try:
            top_class_idx = prediction_result["top_classes"][0]
            confidence = float(prediction_result["top_scores"][0])

            # Validate class index
            if top_class_idx >= len(self.class_names):
                logger.error(
                    f"Class index out of range: {top_class_idx} >= {len(self.class_names)}"
                )
                predicted_text = "Recognition Error"
                confidence_level = "ERROR"
            elif confidence < settings.CONFIDENCE_THRESHOLD:
                predicted_text = "Recognizing..."
                confidence_level = "LOW"
            elif confidence < 0.8:
                predicted_text = self.class_names[top_class_idx]
                confidence_level = "MEDIUM"
            else:
                predicted_text = self.class_names[top_class_idx]
                confidence_level = "HIGH"

            # Create top predictions with index validation
            top_predictions = []
            for idx, score in zip(
                prediction_result["top_classes"], prediction_result["top_scores"]
            ):
                if idx < len(self.class_names):
                    top_predictions.append(
                        {"class": self.class_names[idx], "confidence": float(score)}
                    )

            return {
                "session_id": session_id,
                "frame_index": frame_index,
                "predicted_class": predicted_text,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "top_predictions": top_predictions,
                "has_hand_detection": keypoints["hand_landmarks"]["left_hand"]
                is not None
                or keypoints["hand_landmarks"]["right_hand"] is not None,
                "has_pose_detection": keypoints["pose_landmarks"] is not None,
            }

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return self._create_error_response(session_id, frame_index, str(e))

    def _create_no_detection_response(
        self, session_id: str, frame_index: int
    ) -> Dict[str, Any]:
        """Create response for no detection"""
        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "predicted_class": "No hand gesture detected",
            "confidence": 0.0,
            "confidence_level": "NONE",
            "top_predictions": [],
            "has_hand_detection": False,
            "has_pose_detection": False,
        }

    def _create_error_response(
        self, session_id: str, frame_index: int, error_msg: str
    ) -> Dict[str, Any]:
        """Create error response"""
        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "predicted_class": "Processing error occurred",
            "confidence": 0.0,
            "confidence_level": "ERROR",
            "top_predictions": [],
            "has_hand_detection": False,
            "has_pose_detection": False,
            "error": error_msg,
        }

    def __del__(self):
        """Clean up resources"""
        try:
            if hasattr(self, "keypoint_extractor"):
                del self.keypoint_extractor
        except Exception as e:
            logger.debug(f"Predictor cleanup warning: {e}")
