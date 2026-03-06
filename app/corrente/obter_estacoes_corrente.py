from app.corrente.corrente import Corrente
from app.contexto.contexto import Contexto
from app.servico.api.citibikes_api import CitiBikesAPI

class ObterEstacoesCorrente(Corrente):
    def __init__(self):
        super().__init__()
        self.__api = CitiBikesAPI()

    def executar_processo(self, contexto: Contexto) -> bool:
        estacoes = self.__api.obter_dados()
        contexto.lista_estacoes = estacoes
        return True
