import importlib.util
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_QA_EM_FORMAT_PATH = _REPO_ROOT / "verl" / "utils" / "reward_score" / "qa_em_format.py"
_SPEC = importlib.util.spec_from_file_location("qa_em_format", _QA_EM_FORMAT_PATH)
qa_em_format = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qa_em_format)
