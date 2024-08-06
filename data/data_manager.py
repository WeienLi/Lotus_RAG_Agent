import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chromaManager import ChromaManager
from src.utils.ollamaManager import OllamaManager


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_db(manager, dir_path, ignore_range):
    manager.create_collection()
    manager.load_and_store_data(dir_path=dir_path, ignore_range=ignore_range)


def remove_collection(manager):
    manager.create_collection()
    manager.remove_all()


def reassign_ids(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        entry['id'] = i

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        print('Saved {} entries'.format(len(data)))


if __name__ == '__main__':
    # reassign_ids('./final_data/lotus/car_stats.json')

    config_path = "../src/config/config.yaml"
    config = load_config(config_path)
    collection0_dir = "./final_data/lotus"  # lotus, all data
    collection1_dir = "./final_data/lotus_car_stats"  # lotus_car_stats
    collection2_dir = "./final_data/lotus_brand_info"  # lotus_brand_info
    # collection_dir_lst = [collection0_dir]
    collection_dir_lst = [collection0_dir, collection1_dir, collection2_dir]
    for collection_dir in collection_dir_lst:
        manager = ChromaManager(
            config=config,
            collection_name=collection_dir.split('/')[-1],
        )
        # remove_collection(manager)
        load_db(manager, dir_path=collection_dir, ignore_range=False)
