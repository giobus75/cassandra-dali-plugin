#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
    cd examples/lightning 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --table-suffix=train_256_jpg 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --table-suffix=val_256_jpg 
    rm -f ids_cache/* 
    python3 cache_uuids.py --keyspace=imagenette --table-suffix=train_256_jpg 
    python3 cache_uuids.py --keyspace=imagenette --table-suffix=val_256_jpg 
    python3 train_model.py --num-gpu 1 -a resnet50 --b 128 --workers 4 --lr=0.4 --keyspace=imagenette --train-table-suffix=train_256_jpg --val-table-suffix=val_256_jpg --epochs 1
   `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"