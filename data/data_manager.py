import os
import sys
import yaml
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.chromaManager import ChromaManager
from src.utils.ollamaManager import OllamaManager

"""
data_manager.py

This module is has base functionality for database operations as well as cleaning json file ID.
"""

def load_config(config_path):
    """Safely Load the YAML configuration file

    Args:
        config_path (str): The Path towards the yaml configuration File

    Returns:
        Dict: Parsed YAML content as a Dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_db(manager, dir_path, ignore_range):
    """Populate the current chromadb with data in dir_path.

    Args:
        manager (chromaManager): A chromaManager instance that will be used to load data. 
        dir_path (str): The directory path in string towards the data files in JSON
        ignore_range (bool, optional): A Boolean whether we would like to ignore page range Defaults to False
    """
    manager.create_collection()
    manager.load_and_store_data(dir_path=dir_path, ignore_range=ignore_range)


def remove_collection(manager):
    """Intialize the chroma db of manager instance and remove all the data stored.

    Args:
        manager (chromaManager): A chromaManager Instance that will be used to remove data.
    """
    manager.create_collection()
    manager.remove_all()


def reassign_ids(file_path):
    """Reassigns sequential IDs to each entry in a JSON file.

    Args:
        file_path (str): The path towards json files
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        entry['id'] = i

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        print('Saved {} entries'.format(len(data)))


if __name__ == '__main__':
    
    config_path = "./src/config/config.yaml"
    config = load_config(config_path)
    collection0_dir = config["all_data"]  # lotus, all data
    collection1_dir = config["car_stats"]  # lotus_car_stats
    collection2_dir = config["brand_info"]  # lotus_brand_info
    collection_dir_lst = [collection0_dir, collection1_dir, collection2_dir]
    for collection_dir in collection_dir_lst:
        manager = ChromaManager(
            config=config,
            collection_name=collection_dir.split('/')[-1],
        )
        # remove_collection(manager)
        load_db(manager, dir_path=collection_dir, ignore_range=False)
