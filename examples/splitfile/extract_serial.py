# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        # Read Cassandra parameters
        from private_data import cass_conf

        extract_common.send_images_to_db(
            cass_conf=cass_conf,
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