# label data loader -> cache 방식, S3에서 가져오는 방식 구현
import json
import os
from typing import List, Dict, Any, Union
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class LabelLoader:
    def __init__(self):
        self._labels_cache = None
    
    async def load_class_labels(self)
