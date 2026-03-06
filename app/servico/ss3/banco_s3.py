import json
from typing import List, Dict

import duckdb
from datetime import datetime

from app.servico.ss3.IServicoS3 import IServicoS3


class BancoS3(IServicoS3):

    def __init__(self):
        self.__con = duckdb.connect()
        self.__con.execute(
            "INSTALL httpfs"
        )
        self.__con.execute(
            "LOAD httpfs"
        )
        #           SET s3_endpoint='localhost:9000';
        self.__con.execute(
            """
          
 
            SET s3_endpoint='minio:9000';
            SET s3_access_key_id='minio';
            SET s3_secret_access_key='minio123';
            SET s3_region='us-east-1';
            SET s3_url_style='path';
            SET s3_use_ssl=false;
            """
        )

    def guardar_dados(self, dados):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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