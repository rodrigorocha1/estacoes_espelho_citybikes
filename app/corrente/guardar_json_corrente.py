from app.contexto.contexto import Contexto
from app.corrente.corrente import Corrente
from app.servico.ss3.IServicoS3 import IServicoS3


class GuardarJsonCorrente(Corrente):

    def __init__(self, con_s3: IServicoS3):
        super().__init__()
        self.__con_s3 = con_s3

    def executar_processo(self, contexto: Contexto) -> bool:
        dados = contexto.lista_estacoes
        self.__con_s3.guardar_dados(dados)
        return True
