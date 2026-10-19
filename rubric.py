import json
import re
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Union
import json_repair
import numpy as np
import torch
from utils import MODEL_CHECKPOINTS


class Rubric:
    def __init__(self, rubric_json_or_str: Union[str, Dict[str, Any], List[Any]]):
        self.rubric: str = json.dumps(rubric_json_or_str) if isinstance(rubric_json_or_str, (dict, list)) else str(rubric_json_or_str)
        self.rubric_json: Union[Dict[str, Any], List[Any], None] = None
        self.dimensions: List[str] = self.get_dimensions_from_rubric()

    def get_rubric_in_str(self) -> str:
        """Returns all formatted dimension blocks joined by double newlines."""
        return "\n\n".join(self.dimensions)

    def get_rubric_in_json(self) -> Union[Dict[str, Any], List[Any], None]:
        """Returns the parsed JSON dictionary/list structure if available."""
        if self.rubric_json is not None:
            return self.rubric_json

        try:
            cleaned = self._sanitize_raw_string(self.rubric)
            self.rubric_json = json_repair.loads(cleaned)
            return self.rubric_json
        except Exception:
            return None

    def get_dimensions_embedding(self, encoder) -> np.ndarray:
        """Encodes rubric dimension strings into L2-normalized numpy embeddings."""
        if not self.dimensions:
            return np.empty((0, 0))

        embeddings = encoder.encode(self.dimensions, normalize_embeddings=True)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings
    
    def rubric_exists(self):
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"ablation_results.csv"
        if out_path.exists():
            rubric_results = pd.read_csv(out_path)
            for index, row in rubric_results.iterrows():
                # construct file path
                llm,seed,temp,length,nsamples,threshold,rubric,normalizer,gate,similarity,novelty,retention,beta,alpha,strategy,run_duration_sec,bert,bleu,sbleu,rouge1,rougeL,meteor = rubric_results.columns.tolist()
                model_checkpoint = MODEL_CHECKPOINTS[row[llm]]
                llm = model_checkpoint.replace('/', '-').replace('.', '-')
                path = output_dir / f"rubrics-by-{llm}-{row[seed]}-{row[temp]}-{row[length]}-{row[nsamples]}-{row[threshold]}-{row[rubric]}-{row[normalizer]}-{row[gate]}-{row[similarity]}-{row[novelty]}-{row[retention]}-{row[beta]}-{row[alpha]}-{row[strategy]}.csv"
                # Load all the rubrics
                rubrics = pd.read_csv(path)
                # select the last entry
                content = rubrics.iloc[-1]['rubric']
                # Jsonify the input
                content_json = Rubric(content).get_rubric_in_json()
                
                if self.rubric_json == content_json:
                    return True, path
            
        return False, ""

    @staticmethod
    def _sanitize_raw_string(raw_str: str) -> str:
        """Strips non-breaking spaces and markdown code blocks."""
        if not raw_str:
            return ""
        cleaned = raw_str.replace("\xa0", " ").strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE
        ).strip()
        return cleaned

    def get_dimensions_from_rubric(self) -> List[str]:
        """Parses a rubric string (JSON or Plain Text) into a list of dimension text blocks."""
        if not self.rubric or not self.rubric.strip():
            return []

        parsed_units: List[str] = []
        cleaned_rubric = self._sanitize_raw_string(self.rubric)

        # -------------------------------------------------------------
        # 1. Primary Attempt: Parse using json_repair
        # -------------------------------------------------------------
        try:
            # json_repair.loads automatically handles truncated keys, unclosed quotes/brackets, etc.
            data = json_repair.loads(cleaned_rubric)
            rubric_items = (
                data.get("rubric", []) if isinstance(data, dict) else data
            )

            if isinstance(rubric_items, list):
                for item in rubric_items:
                    if not isinstance(item, dict):
                        continue

                    dim_name = item.get("dimension") or item.get("Dimension")
                    scores = item.get("scores") or item.get("score")

                    if (
                        not dim_name
                        or not scores
                        or not isinstance(dim_name, str)
                    ):
                        continue

                    cleaned_guides = []

                    # Dict format: {"1": "text", "2": "text"}
                    if isinstance(scores, dict):
                        for k, desc in scores.items():
                            if (
                                str(k).isdigit()
                                and int(k) in {1, 2, 3, 4, 5}
                                and desc
                            ):
                                score_num = int(k)
                                cleaned_guides.append(
                                    (
                                        score_num,
                                        f"{score_num}: {str(desc).strip()}",
                                    )
                                )

                    # List format fallback
                    elif isinstance(scores, list):
                        for idx, item_val in enumerate(scores):
                            if isinstance(item_val, str):
                                score_num = idx + 1
                                if 1 <= score_num <= 5:
                                    cleaned_guides.append(
                                        (
                                            score_num,
                                            f"{score_num}: {item_val.strip()}",
                                        )
                                    )
                            elif isinstance(item_val, dict):
                                score_num = item_val.get("score")
                                desc = item_val.get(
                                    "description"
                                ) or item_val.get("criteria", "")
                                if (
                                    isinstance(score_num, str)
                                    and score_num.isdigit()
                                ):
                                    score_num = int(score_num)
                                if score_num in {1, 2, 3, 4, 5} and desc:
                                    cleaned_guides.append(
                                        (score_num, f"{score_num}: {desc.strip()}")
                                    )

                    if cleaned_guides:
                        cleaned_guides.sort(key=lambda x: x[0])
                        guides_text = "\n".join([g[1] for g in cleaned_guides])
                        unit_str = f"{dim_name.strip()}\n{guides_text}"
                        parsed_units.append(unit_str)

                if parsed_units:
                    self.rubric_json = data
                    self.dimensions = parsed_units
                    return self.dimensions

        except Exception:
            pass

        # -------------------------------------------------------------
        # 2. Fallback Attempt: Plain-Text Parsing
        # -------------------------------------------------------------
        blocks = self.rubric.strip().split("\n\n")

        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if len(lines) < 2:
                continue

            dim_header = lines[0].strip("#* {}[]\"'")
            if not dim_header or dim_header in {":", ",", "rubric"}:
                continue

            cleaned_guides = []
            found_scores = set()

            for line in lines[1:]:
                if ":" not in line:
                    continue
                score_part, content_part = line.split(":", 1)

                match = re.search(r"\b([1-5])\b", score_part)
                if match:
                    score_val = int(match.group(1))
                    found_scores.add(score_val)
                    cleaned_text = content_part.strip(' ",\'')
                    if cleaned_text:
                        cleaned_guides.append(
                            (score_val, f"{score_val}: {cleaned_text}")
                        )

            if found_scores:
                cleaned_guides.sort(key=lambda x: x[0])
                guides_text = "\n".join([g[1] for g in cleaned_guides])
                unit_str = f"{dim_header}\n{guides_text}"
                parsed_units.append(unit_str)

        self.dimensions = parsed_units
        return self.dimensions