from datetime import timedelta

import pandas as pd
import pytz

from app.servico.ss3.banco_s3 import BancoS3

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:.2f}'.format)

bs = BancoS3()


caminho = "s3://citybikes/*/*.json"

dados = bs.consultar_dados(id_consulta='1=1', caminho_consulta=caminho)
df_normalized = pd.json_normalize(dados['value'])


df_normalized['extra.last_updated'] = pd.to_datetime(df_normalized['extra.last_updated'], utc=True)


df_normalized['timestamp_brt'] = df_normalized['extra.last_updated'].dt.tz_convert('America/Sao_Paulo')

df_normalized['timestamp_brt_local'] = df_normalized['timestamp_brt'].dt.tz_localize(None)

bs.guardar_dados(df_normalized)