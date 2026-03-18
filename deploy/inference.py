import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

# Disable network checks at module level to ensure it's set before any imports
# This prevents "Checking connectivity to the model hosters" delays
# PaddleX uses PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK (not DISABLE_MODEL_SOURCE_CHECK)
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # For transformers/HuggingFace
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'  # For PaddleX

import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import HfArgumentParser, PreTrainedTokenizer, ProcessorMixin, set_seed

from data.data_utils import (
    box_two_point_convert,
    normalize_bbox,
    sort_boxes,
    string_f2h,
)
from model import PEneoConfig, PEneoModel
from model.backbone_mapping import BACKBONE_MAPPING
from model.peneo_decoder import HandshakingTaggingScheme
from pipeline.decode import sample_decode_peneo

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    backbone_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models",
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    visualize_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the visualization of the prediction"},
    )
    layout_model_dir: Optional[str] = field(
        default=None,  # Will be set via ModelArguments if provided
        metadata={
            "help": "Path to nhance layout detection model directory (PP-DocLayout_plus-L). "
            "If provided, will filter entity linking to only occur within table boundaries."
        },
    )
    layout_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Threshold for layout detection model (default: 0.5)"
        },
    )


@dataclass
class DataArguments:
    dir_image: str = field(
        default=None,
        metadata={"help": "Path to image file or directory containing images"},
    )

    dir_ocr: str = field(
        default=None,
        metadata={
            "help": "Path to OCR JSON file or directory containing OCR JSON files. "
            "OCR JSON format: Array of objects with 'text'/'ocr' and 'bbox'/'box' fields. "
            "Example: [{\"text\": \"...\", \"bbox\": [x1, y1, x2, y2]}, ...] "
            "Or: {\"texts\": [{\"text\": \"...\", \"bbox\": [...]}, ...]}. "
            "Required if apply_ocr=False. If apply_ocr=True, OCR will be run automatically."
        },
    )

    apply_ocr: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply OCR to the image or not, by default False. "
            "If True, will use the built-in Tesseract OCR engine of transformers processor "
            "and ignore provided OCR JSON files. If False, dir_ocr must be provided."
        },
    )

    score_thresh: float = field(
        default=0.0,
        metadata={
            "help": "The score threshold when decoding the prediction matrix. "
            "Default 0.0 for balanced precision-recall."
        },
    )


def visualize(dir_image: str, pred_results: List[Tuple], dir_save: str):
    image = Image.open(dir_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("deploy/Deng.ttf", 10)

    pred_kv_results, pred_line_results = pred_results
    for key_text, value_text, key_box, value_box in pred_kv_results:
        key_left, key_top, key_right, key_bottom = key_box
        value_left, value_top, value_right, value_bottom = value_box

        draw.rectangle(key_box, outline="red", width=2)
        draw.rectangle(value_box, outline="green", width=2)

        draw.text((key_left, key_top - 12), key_text, fill="red", font=font)
        draw.text((value_left, value_top - 12), value_text, fill="green", font=font)

        draw.line(
            [(key_right, key_bottom), (value_left, value_top)], fill="blue", width=2
        )

    for line_text, line_box in pred_line_results:
        line_left, line_top, line_right, line_bottom = line_box
        draw.rectangle(
            [line_left + 2, line_top + 2, line_right - 2, line_bottom - 2],
            outline="gray",
            width=1,
        )

    image.save(dir_save)


class InferenceService:
    def __init__(self, model_args: ModelArguments, data_args: DataArguments):
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        
        # Time PEneo model loading
        peneo_start = time.time()
        self.config: PEneoConfig = PEneoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path
            ),
        )
        self.config.inference_mode = True
        self.backbone_info = BACKBONE_MAPPING.get(self.config.backbone_name, None)

        if self.backbone_info is None:
            logger.error(
                f"Invalid backbone name {self.config.backbone_name}",
                f"Available backbones are {list(BACKBONE_MAPPING.keys())}",
            )
            raise ValueError()

        self.apply_ocr = data_args.apply_ocr
        if self.apply_ocr:
            logger.info(
                "apply_ocr is set to True, using built-in Tesseract OCR engine of transformers processor. The provided OCR path will be ignored.",
            )
        self.processor = self.backbone_info.processor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            apply_ocr=data_args.apply_ocr,
        )
        if isinstance(self.processor, ProcessorMixin):
            self.tokenizer: PreTrainedTokenizer = self.processor.tokenizer
            self.image_processor = self.processor.image_processor
            self.require_image = True
        else:
            self.tokenizer: PreTrainedTokenizer = self.processor
            self.image_processor = None
            self.require_image = False
        self.tokenizer_fetcher = self.backbone_info.tokenizer_fetcher

        self.max_token_len = self.backbone_info.max_token_len
        self.add_cls_token = self.backbone_info.add_cls_token
        self.add_sep_token = self.backbone_info.add_sep_token
        self.padding_side = self.tokenizer.padding_side

        self.model = PEneoModel.from_pretrained(
            model_args.model_name_or_path,
            config=self.config,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.eval()

        self.handshaking_tagger = HandshakingTaggingScheme()
        self.score_thresh = data_args.score_thresh
        peneo_end = time.time()
        print(f"  PEneo model loading: {peneo_end - peneo_start:.2f}s")
        
        logger.info(f"Model loaded from {model_args.model_name_or_path}")
        logger.info(f"Score threshold: {self.score_thresh}")

        # Initialize layout detection model for table filtering
        self.layout_model = None
        self.layout_model_dir = model_args.layout_model_dir
        self.layout_threshold = model_args.layout_threshold
        if self.layout_model_dir:
            try:
                # Time the PaddleX import and model loading separately
                paddlex_import_start = time.time()
                from paddlex import create_model
                paddlex_import_end = time.time()
                print(f"  PaddleX import time: {paddlex_import_end - paddlex_import_start:.2f}s")
                
                layout_start = time.time()
                logger.info(f"Loading layout detection model from {self.layout_model_dir}")
                self.layout_model = create_model(
                    model_name="PP-DocLayout_plus-L",
                    model_dir=self.layout_model_dir
                )
                layout_end = time.time()
                print(f"  Layout detection model loading: {layout_end - layout_start:.2f}s")
                logger.info("Layout detection model loaded successfully. Entity linking will be filtered to table boundaries.")
            except Exception as e:
                logger.warning(f"Failed to load layout detection model: {e}. Table filtering will be disabled.")
                self.layout_model = None

        self.NO_BATCH_KEYS = []
        self.NO_TENSOR_KEYS = [
            "text",
            "relations",
            "line_extraction_shaking_tag",
            "ent_linking_head_rel_shaking_tag",
            "ent_linking_tail_rel_shaking_tag",
            "line_grouping_head_rel_shaking_tag",
            "line_grouping_tail_rel_shaking_tag",
        ]

        self.visualize_path = model_args.visualize_path

    def _detect_tables(self, image_path: str) -> List[List[float]]:
        """
        Detect table regions in the image using layout detection model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of table bounding boxes in format [x1, y1, x2, y2]
        """
        if not self.layout_model:
            return []
        
        try:
            output = self.layout_model.predict(
                image_path,
                batch_size=1,
                layout_nms=True,
                threshold=self.layout_threshold
            )
            
            table_boxes = []
            for res in output:
                # Extract boxes from result
                if hasattr(res, 'res'):
                    res_data = res.res
                else:
                    res_data = res
                
                if isinstance(res_data, dict) and 'boxes' in res_data:
                    boxes = res_data['boxes']
                elif hasattr(res, 'boxes'):
                    boxes = res.boxes
                else:
                    continue
                
                # Filter for 'table' category (cls_id=4 based on your res.json)
                for box in boxes:
                    if isinstance(box, dict):
                        label = box.get('label', '').lower()
                        coordinate = box.get('coordinate', [])
                    else:
                        label = getattr(box, 'label', '').lower()
                        coordinate = getattr(box, 'coordinate', [])
                    
                    if label == 'table' and len(coordinate) == 4:
                        # coordinate is [x1, y1, x2, y2]
                        table_boxes.append([float(c) for c in coordinate])
            
            if table_boxes:
                logger.info(f"Detected {len(table_boxes)} table(s) in {os.path.basename(image_path)}")
            return table_boxes
            
        except Exception as e:
            logger.warning(f"Error detecting tables: {e}")
            return []
    
    def _is_bbox_within_tables(self, bbox: List[float], table_boxes: List[List[float]]) -> bool:
        """
        Check if a bounding box is within any table boundary.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            table_boxes: List of table bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            True if bbox is within any table, False otherwise
        """
        if not table_boxes:
            return False  # No tables detected, don't filter (allow all)
        
        bx1, by1, bx2, by2 = bbox
        
        for table_box in table_boxes:
            tx1, ty1, tx2, ty2 = table_box
            
            # Check if bbox center is within table
            bbox_center_x = (bx1 + bx2) / 2
            bbox_center_y = (by1 + by2) / 2
            
            # Check if bbox is mostly within table (at least 50% overlap)
            # Calculate intersection
            inter_x1 = max(bx1, tx1)
            inter_y1 = max(by1, ty1)
            inter_x2 = min(bx2, tx2)
            inter_y2 = min(by2, ty2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                # There is intersection
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                bbox_area = (bx2 - bx1) * (by2 - by1)
                
                # If at least 50% of bbox is within table, consider it inside
                if bbox_area > 0 and (inter_area / bbox_area) >= 0.5:
                    return True
        
        return False
    
    def _find_containing_table(self, bbox: List[float], table_boxes: List[List[float]]) -> Optional[int]:
        """
        Find which table (if any) contains the bbox.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            table_boxes: List of table bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Index of containing table, or None if not in any table
        """
        for idx, table_box in enumerate(table_boxes):
            if self._is_bbox_within_tables(bbox, [table_box]):
                return idx
        return None
    
    def _can_link_bboxes(self, bbox1: List[float], bbox2: List[float], table_boxes: List[List[float]]) -> bool:
        """
        Check if two bboxes can be linked based on table boundaries.
        
        Rules:
        - Both bboxes in the same table: Allowed
        - Both bboxes outside all tables: Allowed
        - One in table, one outside: Not allowed (cross-boundary)
        - Both in different tables: Not allowed (cross-table)
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            table_boxes: List of table bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            True if linking is allowed, False otherwise
        """
        if not table_boxes:
            # No tables detected - allow all linking
            return True
        
        # Find which table (if any) contains each bbox
        table_idx1 = self._find_containing_table(bbox1, table_boxes)
        table_idx2 = self._find_containing_table(bbox2, table_boxes)
        
        # Both in same table: Allow
        if table_idx1 is not None and table_idx1 == table_idx2:
            return True
        
        # Both outside tables: Allow
        if table_idx1 is None and table_idx2 is None:
            return True
        
        # One in table, one outside: Not allowed
        # Both in different tables: Not allowed
        return False
    def _special_text_replace(self, line_text: str) -> str:
        line_text = line_text.replace("☐", "")
        line_text = line_text.replace("☑", "")
        line_text = line_text.replace("\uf702", "")
        line_text = line_text.replace("\uf703", "")
        line_text = line_text.replace("Tοpic", "Topic")  # ? Magic, don't remove
        line_text = line_text.replace("á", "a")
        line_text = line_text.replace("é", "e")
        line_text = line_text.replace("í", "i")
        line_text = line_text.replace("ó", "o")
        line_text = line_text.replace("ú", "u")
        line_text = line_text.replace("ü", "u")
        line_text = line_text.replace("–", "-")
        line_text = line_text.replace("‘", "'")
        line_text = line_text.replace("’", "'")
        line_text = line_text.replace("“", '"')
        line_text = line_text.replace("—", "-")
        line_text = line_text.replace("™", "TM")
        line_text = line_text.replace("§", "")
        line_text = line_text.replace("¢", "")

        return string_f2h(line_text)

    def preprocess(
        self,
        image_path: Union[str, List[str]],
        ocr_path: Union[str, List[str]] = None,
    ):
        if isinstance(image_path, str):
            if os.path.isdir(image_path):
                image_path_list = os.listdir(image_path)
                image_path_list = [os.path.join(image_path, x) for x in image_path_list]
            else:
                image_path_list = [image_path]

            image_path_list.sort()
        else:
            raise ValueError("image_path must be a string")

        if not ocr_path:
            assert self.apply_ocr, "OCR path must be provided if apply_ocr is False"
            ocr_path_list = [None] * len(image_path_list)
        else:
            if isinstance(ocr_path, str):
                if os.path.isdir(ocr_path):
                    ocr_path_list = os.listdir(ocr_path)
                    ocr_path_list = [os.path.join(ocr_path, x) for x in ocr_path_list]
                else:
                    ocr_path_list = [ocr_path]
            else:
                raise ValueError("ocr_path must be a string")

            ocr_path_list.sort()

        assert len(image_path_list) == len(
            ocr_path_list
        ), "Number of image and OCR paths must be the same"

        for i, curr_image_path in enumerate(image_path_list):
            image = Image.open(curr_image_path).convert("RGB")
            image_w, image_h = image.size
            if self.require_image or self.apply_ocr:
                image_features = self.image_processor(images=image, return_tensors="pt")
                image_return = image_features["pixel_values"].to(self.device)
            else:
                image_return = None

            curr_ocr_path = ocr_path_list[i]
            if self.apply_ocr or not curr_ocr_path or not os.path.exists(curr_ocr_path):
                # Use built-in OCR from image processor
                if not self.apply_ocr:
                    logger.warning(
                        f"OCR path not provided or file not found: {curr_ocr_path}. "
                        f"Using built-in OCR from image processor."
                    )
                line_text_list = image_features["words"][0]
                line_box_list = image_features["boxes"][0]
            else:
                # Load OCR JSON file
                # Expected format: [{"text": "...", "bbox": [x1, y1, x2, y2]}, ...]
                # Or: {"texts": [{"text": "...", "bbox": [...]}, ...]}
                # Bbox can be 4-point [x1, y1, x2, y2] or 8-point polygon
                # Text field can be "text" or "ocr", bbox field can be "bbox" or "box"
                try:
                    curr_ocr = json.load(open(curr_ocr_path, "r", encoding="utf-8"))
                    if "texts" in curr_ocr:
                        curr_ocr = curr_ocr["texts"]
                    line_text_list, line_box_list = [], []
                    for line_info in curr_ocr:
                        # Extract text (prefer "ocr" field, fallback to "text")
                        if "ocr" in line_info:
                            line_text_list.append(line_info["ocr"])
                        elif "text" in line_info:
                            line_text_list.append(line_info["text"])
                        else:
                            logger.warning(f"OCR entry missing text field: {line_info}")
                            line_text_list.append("")
                        
                        # Extract bbox (prefer "bbox" field, fallback to "box")
                        if "bbox" in line_info:
                            line_box_list.append(box_two_point_convert(line_info["bbox"]))
                        elif "box" in line_info:
                            line_box_list.append(box_two_point_convert(line_info["box"]))
                        else:
                            logger.warning(f"OCR entry missing bbox/box field: {line_info}")
                            line_box_list.append([0, 0, 0, 0])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OCR JSON file {curr_ocr_path}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error loading OCR JSON from {curr_ocr_path}: {e}")
                    raise

            ro_sorted_box_idx_list = sort_boxes(line_box_list)

            # Reserve tokens for CLS and SEP tokens
            reserved_tokens = 0
            if self.add_cls_token:
                reserved_tokens += 1
            if self.add_sep_token:
                reserved_tokens += 1
            available_tokens = self.max_token_len - reserved_tokens

            # Pre-compute token lengths for all lines
            line_token_lengths = []
            for ro_sorted_idx in ro_sorted_box_idx_list:
                line_text = line_text_list[ro_sorted_idx]
                line_text = self._special_text_replace(line_text)
                line_tokens = self.tokenizer.tokenize(line_text)
                line_token_ids = self.tokenizer.convert_tokens_to_ids(line_tokens)
                line_token_lengths.append((ro_sorted_idx, len(line_token_ids)))

            # Split into chunks
            total_lines = len(ro_sorted_box_idx_list)
            chunks = []
            chunk_start = 0
            chunk_token_count = 0
            
            for idx, (ro_sorted_idx, line_token_len) in enumerate(line_token_lengths):
                if line_token_len == 0:
                    continue
                
                # Check if adding this line would exceed the limit
                if chunk_token_count + line_token_len > available_tokens and chunk_token_count > 0:
                    # Save current chunk and start new one
                    chunks.append((chunk_start, idx))
                    chunk_start = idx
                    chunk_token_count = line_token_len
                else:
                    chunk_token_count += line_token_len
            
            # Add final chunk
            if chunk_start < len(line_token_lengths):
                chunks.append((chunk_start, len(line_token_lengths)))
            
            if len(chunks) > 1:
                logger.info(
                    f"📄 Document split into {len(chunks)} chunks: "
                    f"{total_lines} total lines, "
                    f"max {available_tokens} tokens per chunk"
                )

            # Process each chunk
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                texts = []
                input_ids = []
                bbox = []
                orig_bbox = []
                
                # Process lines in this chunk
                for idx in range(chunk_start, chunk_end):
                    ro_sorted_idx, line_token_len = line_token_lengths[idx]
                    line_text = line_text_list[ro_sorted_idx]
                    line_text = self._special_text_replace(line_text)
                    line_tokens = self.tokenizer.tokenize(line_text)
                    line_token_ids = self.tokenizer.convert_tokens_to_ids(line_tokens)
                    
                    if len(line_tokens) == 0:
                        continue

                    if self.tokenizer_fetcher is not None:
                        line_sos_processed_tokens = self.tokenizer_fetcher(
                            line_text, line_tokens
                        )
                    else:
                        line_sos_processed_tokens = line_tokens

                    line_orig_bbox = line_box_list[ro_sorted_idx]
                    line_norm_bbox = normalize_bbox(line_orig_bbox, (image_w, image_h))

                    orig_bbox.extend([line_orig_bbox] * line_token_len)
                    bbox.extend([line_norm_bbox] * line_token_len)
                    texts.extend(line_sos_processed_tokens)
                    input_ids.extend(line_token_ids)

                if self.add_cls_token:
                    input_ids = [self.tokenizer.cls_token_id] + input_ids
                    orig_bbox = [[0, 0, 0, 0]] + orig_bbox
                    bbox = [[0, 0, 0, 0]] + bbox
                if self.add_sep_token:
                    input_ids.append(self.tokenizer.sep_token_id)
                    orig_bbox.append([0, 0, 0, 0])
                    bbox.append([0, 0, 0, 0])

                features = {
                    "fname": [curr_image_path],
                    "image_path": [curr_image_path],
                    "input_ids": [input_ids],
                    "bbox": [bbox],
                    "orig_bbox": [orig_bbox],
                    "text": [texts],
                }
                batch = self.tokenizer.pad(
                    features,
                    padding="longest",
                    max_length=512,
                    pad_to_multiple_of=8,
                    return_tensors=None,
                )
                sequence_length = torch.tensor(batch["input_ids"]).shape[1]
                if self.padding_side == "right":
                    batch["bbox"] = [
                        bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                        for bbox in batch["bbox"]
                    ]
                    batch["orig_bbox"] = [
                        bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                        for bbox in batch["orig_bbox"]
                    ]
                else:
                    batch["bbox"] = [
                        [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                        for bbox in batch["bbox"]
                    ]
                    batch["orig_bbox"] = [
                        [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                        for bbox in batch["orig_bbox"]
                    ]

                for k, v in batch.items():
                    if isinstance(v[0], list):
                        if k not in self.NO_BATCH_KEYS and k not in self.NO_TENSOR_KEYS:
                            batch[k] = torch.tensor(v, dtype=torch.int64).to(self.device)
                        elif k not in self.NO_TENSOR_KEYS:
                            batch[k] = [
                                torch.tensor(vv, dtype=torch.int64).to(self.device)
                                for vv in v
                            ]
                        else:
                            batch[k] = v
                    elif isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                    else:
                        batch[k] = v

                to_model = {
                    "input_ids": batch["input_ids"],
                    "bbox": batch["bbox"],
                    "orig_bbox": batch["orig_bbox"],
                    "attention_mask": batch["attention_mask"],
                }
                to_others = {
                    "text": batch["text"],
                    "fname": batch["fname"],
                    "chunk_info": (chunk_idx, len(chunks), chunk_start, chunk_end)  # Track chunk info
                }
                if image_return is not None:
                    to_model.update({"image": image_return})

                yield to_model, to_others

    @torch.no_grad()
    def inference(self, inputs):
        return self.model(**inputs)

    def postprocess(self, score_thresh=None, line_extraction_score_thresh=None, table_boxes=None, **kwargs):
        if score_thresh is None:
            score_thresh = self.score_thresh
        # Use a lower threshold for line extraction to catch widgets
        # Widgets may have lower confidence but should still be extracted
        if line_extraction_score_thresh is None:
            # Default: 2.0 points lower than entity linking threshold
            line_extraction_score_thresh = score_thresh - 2.0 if score_thresh < 0 else max(score_thresh - 0.5, -2.0)
        
        # Add table filtering if table_boxes provided
        decode_kwargs = kwargs.copy()
        if table_boxes is not None:
            decode_kwargs['table_boxes'] = table_boxes
            decode_kwargs['can_link_bboxes_fn'] = self._can_link_bboxes
        
        return sample_decode_peneo(
            handshaking_tagger=self.handshaking_tagger,
            decode_gt=False,
            score_thresh=score_thresh,
            line_extraction_score_thresh=line_extraction_score_thresh,
            **decode_kwargs,
        )[
            :2
        ]  # only return parsed kv-pairs & lines

    def run(self, data_args: DataArguments):
        total_start = time.time()
        
        print("="*60)
        print("Starting PEneo Inference")
        print("="*60)
        
        preprocessing_start = time.time()
        data_getter = self.preprocess(data_args.dir_image, data_args.dir_ocr)
        preprocessing_end = time.time()
        print(f"Preprocessing time: {preprocessing_end - preprocessing_start:.2f}s")
        
        predictions = []
        inference_start = time.time()
        sample_cnt = 0
        
        # Track current image path for table detection (to avoid re-detecting for chunks)
        current_image_path = None
        cached_table_boxes = []
        
        # Track KV pairs and lines per image (to aggregate chunks)
        image_kv_pairs = {}  # key: fname, value: list of KV pairs
        image_lines = {}  # key: fname, value: list of lines
        
        for inputs in tqdm.tqdm(data_getter):
            model_inputs, other_inputs = inputs
            (
                line_extraction_shaking_outputs,
                ent_linking_h2h_shaking_outputs,
                ent_linking_t2t_shaking_outputs,
                line_grouping_h2h_shaking_outputs,
                line_grouping_t2t_shaking_outputs,
                orig_bboxes,
            ) = self.inference(model_inputs)
            texts = other_inputs.get("text")
            fnames = other_inputs.get("fname")

            for (
                line_extraction_shaking_output,
                ent_linking_h2h_shaking_output,
                ent_linking_t2t_shaking_output,
                line_grouping_h2h_shaking_output,
                line_grouping_t2t_shaking_output,
                orig_bbox,
                text,
                fname,
            ) in zip(
                line_extraction_shaking_outputs,
                ent_linking_h2h_shaking_outputs,
                ent_linking_t2t_shaking_outputs,
                line_grouping_h2h_shaking_outputs,
                line_grouping_t2t_shaking_outputs,
                orig_bboxes,
                texts,
                fnames,
            ):
                if len(texts) == 0:
                    continue
                
                # Detect tables for this image (only once per image file, reuse for chunks)
                if fname != current_image_path:
                    current_image_path = fname
                    cached_table_boxes = self._detect_tables(fname) if self.layout_model else []
                    # Initialize KV pairs and lines lists for new image
                    if fname not in image_kv_pairs:
                        image_kv_pairs[fname] = []
                    if fname not in image_lines:
                        image_lines[fname] = []
                
                seq_len = len(orig_bbox)
                shaking_ind2matrix_ind = [
                    (ind, end_ind)
                    for ind in range(seq_len)
                    for end_ind in list(range(seq_len))[ind:]
                ]
                curr_pred_kv, curr_pred_line = self.postprocess(
                    line_extraction_shaking=line_extraction_shaking_output,
                    ent_linking_h2h_shaking=ent_linking_h2h_shaking_output,
                    ent_linking_t2t_shaking=ent_linking_t2t_shaking_output,
                    line_grouping_h2h_shaking=line_grouping_h2h_shaking_output,
                    line_grouping_t2t_shaking=line_grouping_t2t_shaking_output,
                    text=text,
                    shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                    bbox=orig_bbox,
                    score_thresh=self.score_thresh,
                    table_boxes=cached_table_boxes,  # Pass table boxes for filtering
                )

                # Aggregate KV pairs and lines per image (for chunks)
                image_kv_pairs[fname].extend(curr_pred_kv)
                image_lines[fname].extend(curr_pred_line)
                predictions.append(curr_pred_kv)
                sample_cnt += 1

        inference_end = time.time()
        avg_inference_time = (inference_end - inference_start) / sample_cnt if sample_cnt > 0 else 0
        inference_time = inference_end - inference_start
        
        print("="*60)
        print("Prediction Timing Summary")
        print("="*60)
        print(f"Model inference time: {inference_time:.2f}s")
        print(f"Average inference time per sample: {avg_inference_time:.2f}s")
        print(f"Total samples processed: {sample_cnt}")
        
        # Measure postprocessing time (visualization and final processing)
        postprocessing_start = time.time()

        # Visualize accumulated results per image (after all chunks processed)
        if self.visualize_path is not None:
            dir_visualize_save_root = self.visualize_path
            if not os.path.exists(dir_visualize_save_root):
                os.makedirs(dir_visualize_save_root)
            
            for fname, kv_pairs in image_kv_pairs.items():
                lines = image_lines.get(fname, [])
                dir_visualize_save = os.path.join(
                    dir_visualize_save_root, os.path.basename(fname)
                )
                logger.info(f"Visualizing {len(kv_pairs)} KV pairs and {len(lines)} lines for {os.path.basename(fname)}")
                visualize(
                    dir_image=fname,
                    pred_results=(kv_pairs, lines),
                    dir_save=dir_visualize_save,
                )

        # Print final dictionary with widget, label, and bboxes for each image
        for fname, kv_pairs in image_kv_pairs.items():
            mappings = {}
            for kv_pair in kv_pairs:
                if len(kv_pair) >= 4:
                    widget_name = kv_pair[0]
                    label_name = kv_pair[1]
                    widget_bbox = kv_pair[2]
                    label_bbox = kv_pair[3]
                    mappings[widget_name] = {
                        'label': label_name,
                        'widget_bbox': widget_bbox,
                        'label_bbox': label_bbox
                    }
                elif len(kv_pair) >= 2:
                    widget_name = kv_pair[0]
                    label_name = kv_pair[1]
                    mappings[widget_name] = {
                        'label': label_name,
                        'widget_bbox': None,
                        'label_bbox': None
                    }
            
            print(mappings)
        
        postprocessing_end = time.time()
        postprocessing_time = postprocessing_end - postprocessing_start
        
        total_end = time.time()
        total_time = total_end - total_start
        
        print("="*60)
        print("Final Timing Summary")
        print("="*60)
        print(f"Preprocessing time: {preprocessing_end - preprocessing_start:.2f}s")
        print(f"Model inference time: {inference_time:.2f}s")
        print(f"Postprocessing time: {postprocessing_time:.2f}s")
        print(f"Total prediction time: {total_time:.2f}s")
        print("="*60)

        return predictions


def main():
    main_start = time.time()
    
    # Disable network checks BEFORE any model loading to ensure consistent timing
    # (Already set at module level, but ensuring it's set here too)
    os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # For transformers/HuggingFace
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'  # For PaddleX
    
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(42)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    print("="*60)
    print("Model Loading")
    print("="*60)
    model_loading_start = time.time()
    inference_service = InferenceService(model_args, data_args)
    model_loading_end = time.time()
    model_loading_time = model_loading_end - model_loading_start
    print(f"Total model loading time: {model_loading_time:.2f}s")
    print("="*60)
    
    predictions = inference_service.run(data_args)

    logger.info("Predictions:")
    for pred in predictions:
        logger.info(pred)
    
    main_end = time.time()
    print("="*60)
    print(f"Total execution time: {main_end - main_start:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()
