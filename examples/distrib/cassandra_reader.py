# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# cassandra
from cassandra.auth import PlainTextAuthProvider
from crs4.cassandra_utils import MiniListManager

# load cassandra-dali-plugin
import crs4.cassandra_utils
import nvidia.dali.plugin_manager as plugin_manager
import nvidia.dali.fn as fn
import pathlib

# varia
import getpass
import os
import pickle

plugin_path = pathlib.Path(crs4.cassandra_utils.__path__[0])
plugin_path = plugin_path.parent.parent.joinpath("libcrs4cassandra.so")
plugin_path = str(plugin_path)
plugin_manager.load_library(plugin_path)


def get_cassandra_reader(
    keyspace,
    table_suffix,
    shard_id=0,
    num_shards=1,
    io_threads=2,
    prefetch_buffers=2,
    name="Reader",
    shuffle_after_epoch=True,
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ips, username, password
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    # set uuids cache directory
    ids_cache = "ids_cache"
    rows_fn = os.path.join(ids_cache, f"{keyspace}_{table_suffix}.rows")

    # Load list of uuids from Cassandra DB...
    ap = PlainTextAuthProvider(username=username, password=password)
    id_col = "patch_id"
    if not os.path.exists(rows_fn):
        lm = MiniListManager(ap, cassandra_ips)
        conf = {
            "table": f"{keyspace}.ids_{table_suffix}",
            "id_col": id_col,
        }
        lm.set_config(conf)
        print("Loading list of uuids from DB... ", end="")
        lm.read_rows_from_db()
        if shard_id == 0:
            if not os.path.exists(ids_cache):
                os.makedirs(ids_cache)
            lm.save_rows(rows_fn)
        stuff = lm.get_rows()
    else:  # ...or from the cached file
        print("Loading list of uuids from cached file... ", end="")
        with open(rows_fn, "rb") as f:
            stuff = pickle.load(f)
    # init and return Cassandra reader
    uuids = stuff["row_keys"]
    uuids = list(map(str, uuids))  # convert uuids to strings
    print(f" {len(uuids)} images")
    table = f"{keyspace}.data_{table_suffix}"
    cassandra_reader = fn.crs4.cassandra(
        name=name,
        uuids=uuids,
        shuffle_after_epoch=shuffle_after_epoch,
        cassandra_ips=cassandra_ips,
        table=table,
        label_col="label",
        data_col="data",
        id_col=id_col,
        username=username,
        password=password,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        num_shards=num_shards,
        shard_id=shard_id,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
        # use_ssl=True,
        # ssl_certificate="node0.cer.pem",
    )
    return cassandra_reader


def get_cassandra_reader_from_splitfile(
    split_fn,  
    split_index=0,
    shard_id=0,
    num_shards=1,
    io_threads=2,
    prefetch_buffers=2,
    name="Reader",
    shuffle_after_epoch=True,
    comm_threads=2,
    copy_threads=2,
    wait_threads=2,
):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ips, username, password
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cassandra_ips = [cassandra_ip]
        username = getpass("Insert Cassandra user: ")
        password = getpass("Insert Cassandra password: ")

    # Read Pickle File
    data = pickle.load(open(split_fn, "rb"))
    clm_table = data['clm_table'] ## ids table
    clm_grouping_cols = data['clm_grouping_cols'] 
    id_col = data['id_col']
    num_classes = data['num_classes']
    label_map = data['label_map']
    table = data['table'] ## data table
    label_col = data['label_col'] # Name of the table column with the outcome label
    data_col = data['data_col'] # Name of the table column with actual data
    row_keys = data['row_keys'] # Numpy array of UUIDs
    split = data['split'] # List of arrays. Each arrays indexes the row_keys array for each split.
    metatable = data['metatable']

    # init and return Cassandra reader
    
    uuids = row_keys[split[split_index]] # Gets only the UUIDs belonging to the specified split
    uuids = list(map(str, uuids))  # convert uuids to strings
    
    print(f" {len(uuids)} images")
    cassandra_reader = fn.crs4.cassandra(
        name=name,
        uuids=uuids,
        shuffle_after_epoch=shuffle_after_epoch,
        cassandra_ips=cassandra_ips,
        table=table,
        label_col=label_col,
        data_col=data_col,
        id_col=id_col,
        username=username,
        password=password,
        prefetch_buffers=prefetch_buffers,
        io_threads=io_threads,
        num_shards=num_shards,
        shard_id=shard_id,
        comm_threads=comm_threads,
        copy_threads=copy_threads,
        wait_threads=wait_threads,
        # use_ssl=True,
        # ssl_certificate="node0.cer.pem",
    )
    return cassandra_reader
