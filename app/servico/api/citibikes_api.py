from typing import List, Dict, Any

import requests


class CitiBikesAPI:

    def __init__(self):
        self.__url = 'https://api.citybik.es/v2/networks/bike-curitiba'

    def obter_dados(self) -> List[Dict[str, Any]]:
        response = requests.get(self.__url)
        return response.json()['network']['stations']


if __name__ == '__main__':
    citibikes = CitiBikesAPI()
    for estacao in citibikes.obter_dados():
        print(estacao)
