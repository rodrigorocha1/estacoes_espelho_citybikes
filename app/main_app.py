import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.utlis.utlis_log import logger
from contexto.contexto import Contexto
from corrente.obter_estacoes_corrente import ObterEstacoesCorrente


def rotina():
    contexto = Contexto()
    p1 = ObterEstacoesCorrente()
    p1.corrente(contexto=contexto)


def desativar_rotina(signum, frame, ):
    logger.info("Encerrando scheduler...")
    scheduler.shutdown()
    sys.exit(0)


scheduler = BlockingScheduler()
scheduler.add_job(
    rotina,
    trigger=IntervalTrigger(seconds=1),
    id="job_15_min",
    replace_existing=True,
    max_instances=1,
    coalesce=True
)
signal.signal(signal.SIGTERM, desativar_rotina)
signal.signal(signal.SIGINT, desativar_rotina)

if __name__ == "__main__":
    logger.info("Scheduler iniciado...")
    scheduler.start()
