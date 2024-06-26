#! /bin/bash -x
set -e

pgrep -f cassandra || /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" 
    cd examples/imagenette/ 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql 
    pgrep -f spark || (/spark/sbin/start-master.sh  && /spark/sbin/start-worker.sh spark://$HOSTNAME:7077) 
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                            --split-subdir=train --data-table=imagenette.data_train_256_jpg \
			    --metadata-table=imagenette.metadata_train_256_jpg 
    /spark/bin/spark-submit --master spark://$HOSTNAME:7077 --conf spark.default.parallelism=10 \
                            --py-files extract_common.py extract_spark.py /tmp/imagenette2-320/ \
                            --split-subdir=val --data-table=imagenette.data_val_256_jpg \
			    --metadata-table=imagenette.metadata_val_256_jpg 
    rm -f ids_cache/* 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_train_256_jpg 
    python3 loop_read.py --data-table imagenette.data_train_256_jpg --metadata-table imagenette.metadata_train_256_jpg 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_val_256_jpg 
    python3 loop_read.py --data-table imagenette.data_val_256_jpg --metadata-table imagenette.metadata_val_256_jpg
    python3 loop_read.py --data-table imagenette.data_train_256_jpg --metadata-table imagenette.metadata_train_256_jpg --use-gpu 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -e "DROP KEYSPACE IF EXISTS imagenette;" 
    SSL_VALIDATE=false /cassandra/bin/cqlsh --ssl -f create_tables.cql
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --data-table imagenette.data_train_256_jpg --metadata-table imagenette.metadata_train_256_jpg
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --data-table imagenette.data_val_256_jpg --metadata-table imagenette.metadata_val_256_jpg
    rm -f ids_cache/* 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_train_256_jpg 
    python3 loop_read.py --data-table imagenette.data_train_256_jpg --metadata-table imagenette.metadata_train_256_jpg 
    python3 cache_uuids.py --metadata-table=imagenette.metadata_val_256_jpg 
    python3 loop_read.py --data-table imagenette.data_val_256_jpg --metadata-table imagenette.metadata_val_256_jpg
    rm -Rf /tmp/imagenette_256_jpg/ 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=train --target-dir=/tmp/imagenette_256_jpg/train 
    python3 extract_serial.py /tmp/imagenette2-320 --split-subdir=val --target-dir=/tmp/imagenette_256_jpg/val 
    python3 loop_read.py --reader=file --file-root=/tmp/imagenette_256_jpg/train 
    torchrun --nproc_per_node=1 distrib_train_from_cassandra.py -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 \
	     --workers 4 --lr=0.4 --opt-level O2 --epochs 1 \
             --train-data-table imagenette.data_train_256_jpg --train-metadata-table imagenette.metadata_train_256_jpg \
             --val-data-table imagenette.data_val_256_jpg --val-metadata-table imagenette.metadata_val_256_jpg
   `### BEGIN COMMENT \
    ### END COMMENT`
    echo "--- OK ---"
