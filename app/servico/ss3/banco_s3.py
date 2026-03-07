import json
import os
from datetime import datetime
from typing import Dict, List, Union

import duckdb
import pandas as pd
from duckdb.experimental.spark import DataFrame

from app.servico.ss3.IServicoS3 import IServicoS3

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:.2f}'.format)


class BancoS3(IServicoS3):

    def __init__(self):
        self.__con = duckdb.connect()
        self.__con.execute(
            "INSTALL httpfs"
        )
        self.__con.execute(
            "LOAD httpfs"
        )
        #           SET s3_endpoint='localhost:9000';         SET s3_endpoint='minio:9000';

        self.__con.execute(
            """
          
 
            SET s3_endpoint='localhost:9000'; 
            SET s3_access_key_id='minio';
            SET s3_secret_access_key='minio123';
            SET s3_region='us-east-1';
            SET s3_url_style='path';
            SET s3_use_ssl=false;
            """
        )

    def guardar_dados(self, dados: Union[List[Dict[str, int]], DataFrame]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if isinstance(dados, list):
            path = f"s3://citybikes/bikes/bikes_{timestamp}.json"

            json_str = json.dumps(dados)

            self.__con.execute("""
                CREATE OR REPLACE TABLE tmp_json AS
                SELECT * FROM json_each(?)
            """, [json_str])

            self.__con.execute(f"""
                COPY tmp_json
                TO '{path}'
                (FORMAT JSON)
            """)

            print("Arquivo salvo:", path)
        else:
            path = f"s3://citybikes/bikes_prata/bikes_prata.csv"
            self.__con.register('df_temp', dados)
            self.__con.execute(
                f'COPY df_temp TO "{path}" (FORMAT CSV, HEADER TRUE,  DELIMITER "|")'
            )

    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> pd.DataFrame:
        extensao = os.path.splitext(caminho_consulta)[1].lower()
        print(f"Extensão do arquivo: {extensao}")

        df = self.__con.execute(f"""
            SELECT *
            FROM read_json_auto('{caminho_consulta}')
            WHERE {id_consulta}
        """).fetchdf()
        return df
