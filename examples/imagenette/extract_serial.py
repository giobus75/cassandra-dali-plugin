# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# To insert in DB, run with, e.g.,
# python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --keyspace=imagenette --split-subdir=train --table-suffix=train_256_jpg

# To save files in a directory, run with, e.g.,
# python3 extract_serial.py /tmp/imagenette2-320 --img-format=JPEG --split-subdir=train --target-dir=/data/imagenette/train_256_jpg


from getpass import getpass
import extract_common
from clize import run


def save_images(
    src_dir,
    *,
    img_format="JPEG",
    keyspace="imagenette",
    table_suffix="train_256_jpg",
    split_subdir="train",
    target_dir=None,
    img_size=256,
):
    """Save resized images to Cassandra DB or directory

    :param src_dir: Input directory for Imagenette
    :param img_format: Format of output images
    :param keyspace: Name of dataset (for the Cassandra table)
    :param table_suffix: Suffix for table names
    :param target_dir: Output directory (when saving to filesystem)
    :param split_subdir: Subdir to be processed
    :param img_size: Target image size
    """
    splits = [split_subdir]
    jobs = extract_common.get_jobs(src_dir, splits)
    if not target_dir:
        try:
            # Read Cassandra parameters
            from private_data import CassConf as CC
        except ImportError:
            cassandra_ip = getpass("Insert Cassandra's IP address: ")
            cassandra_ips = [cassandra_ip]
            username = getpass("Insert Cassandra user: ")
            password = getpass("Insert Cassandra password: ")

        extract_common.send_images_to_db(
            cloud_config=CC.cloud_config,
            cassandra_ips=CC.cassandra_ips,
            cassandra_port=CC.cassandra_port,
            username=CC.username,
            password=CC.password,
            img_format=img_format,
            keyspace=keyspace,
            table_suffix=table_suffix,
            img_size=img_size,
        )(jobs)
    else:
        extract_common.save_images_to_dir(
            target_dir,
            img_format,
            img_size=img_size,
        )(jobs)


# parse arguments
if __name__ == "__main__":
    run(save_images)
