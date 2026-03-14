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

renomear_colunas = {
    'id': 'id',
    'name': 'nome',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'timestamp': 'timestamp',
    'free_bikes': 'bicicletas_livres',
    'empty_slots': 'vagas_livres',
    'extra.uid': 'uid_extra',
    'extra.renting': 'alugando_extra',
    'extra.returning': 'retornando_extra',
    'extra.last_updated': 'ultimo_update_extra',
    'extra.address': 'endereco_extra',
    'extra.post_code': 'cep_extra',
    'extra.payment': 'pagamento_extra',
    'extra.payment-terminal': 'terminal_pagamento_extra',
    'extra.altitude': 'altitude_extra',
    'extra.slots': 'total_vagas_extra',
    'extra.normal_bikes': 'bicicletas_normais_extra',
    'extra.ebikes': 'ebikes_extra',
    'extra.has_ebikes': 'possui_ebikes_extra',
    'extra.virtual': 'virtual_extra',
    'timestamp_brt': 'timestamp_brt',
    'timestamp_brt_local': 'timestamp_brt_local'
}

# Renomeando colunas
df_normalized = df_normalized.rename(columns=renomear_colunas)

print(df_normalized.columns)

bs.guardar_dados(df_normalized)