from abc import ABC
from typing import List, Dict, Union

import pandas as pd
from mypyc.ir.rtypes import abstractmethod
from pandas import DataFrame


class IServicoS3(ABC):
    @abstractmethod
    def guardar_dados(self, dados: Union[List[Dict[str, int]], DataFrame]) -> None:
        pass

    @abstractmethod
    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> pd.DataFrame:
        pass
