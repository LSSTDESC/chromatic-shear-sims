import os
from pathlib import Path


def get_config_name(config_file):
    config_path = Path(config_file)
    return config_path.stem

def get_output_path(output_dir, config_file):
    config_name = get_config_name(config_file)
    output_path = os.path.join(
        output_dir,
        config_name,
    )
    return output_path

def get_run_path(output_dir, config_file, seed):
    output_path = get_output_path(output_dir, config_file)
    return output_path
    # return os.path.join(
    #     output_path,
    #     f"{seed}.parquet",
    # )

def get_aggregate_dataset(output_dir, config_file):
    config_name = get_config_name(config_file)
    aggregate_name = f"{config_name}_aggregates"
    aggregate_path = os.path.join(
        output_dir,
        aggregate_name,
    )
    return aggregate_path


def get_aggregate_path(output_dir, config_file):
    config_name = get_config_name(config_file)
    aggregate_name = f"{config_name}_aggregates.arrow"
    aggregate_path = os.path.join(
        output_dir,
        aggregate_name,
    )
    return aggregate_path
