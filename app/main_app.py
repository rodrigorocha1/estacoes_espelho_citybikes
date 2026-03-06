import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.corrente.guardar_json_corrente import GuardarJsonCorrente
from app.servico.ss3.banco_s3 import BancoS3
from app.utlis.utlis_log import logger
from app.contexto.contexto import Contexto
from app.corrente.obter_estacoes_corrente import ObterEstacoesCorrente


def rotina():
    contexto = Contexto()
    con_s3 = BancoS3()
    p1 = ObterEstacoesCorrente()
    p2 = GuardarJsonCorrente(con_s3=con_s3)
    p1.set_proxima_corrente(p2)
    p1.corrente(contexto=contexto)


def desativar_rotina(signum, frame, ):
    logger.info("Encerrando scheduler...")
    scheduler.shutdown()
    sys.exit(0)


scheduler = BlockingScheduler()
scheduler.add_job(
    rotina,
    trigger=IntervalTrigger(minutes=15),
    id="job_15_min",
    replace_existing=True,
    max_instances=1,
    coalesce=True
)
signal.signal(signal.SIGTERM, desativar_rotina)
signal.signal(signal.SIGINT, desativar_rotina)

if __name__ == "__main__":
    logger.info("Scheduler iniciado...")
    rotina()
    scheduler.start()
