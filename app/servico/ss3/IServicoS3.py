from abc import ABC
from typing import List, Dict

from mypyc.ir.rtypes import abstractmethod


class IServicoS3(ABC):
    @abstractmethod
    def guardar_dados(self, dados: List[Dict[str, int]]) -> None:
        pass
