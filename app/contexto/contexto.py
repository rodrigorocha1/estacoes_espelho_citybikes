from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Contexto:
    lista_estacoes: List[Dict[str, Any]] = field(default_factory=list)
