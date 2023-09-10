"""Design-time interface for the JobContext
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional


class JobContext(object):

    @property
    def outputPath(self) -> Path:
        raise NotImplementedError
    
    @property
    def data(self) -> Dict[str, Dict]:
        raise NotImplementedError
    
    @property
    def files(self) -> List[Path]:
        raise NotImplementedError

    def log(self, msg: str, progress: Optional[float]=None) -> None:
        raise NotImplementedError

    def fail(self, msg: str) -> None:
        raise NotImplementedError

    def addFile(self, fpath: Path) -> None:
        raise NotImplementedError

    def addInputData(self, name, data: Dict) -> None:
        raise NotImplementedError
