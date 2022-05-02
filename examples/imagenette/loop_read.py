# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandradl import CassandraDataset, CassandraListManager

from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
from time import sleep

# Read Cassandra parameters
try:
    from private_data import cassandra_ip, cass_user, cass_pass
except ImportError:
    cassandra_ip = getpass("Insert Cassandra's IP address: ")
    cass_user = getpass("Insert Cassandra user: ")
    cass_pass = getpass("Insert Cassandra password: ")

# Init Cassandra dataset
ap = PlainTextAuthProvider(username=cass_user, password=cass_pass)

# Create three splits, with ratio 70, 20, 10 and balanced classes
id_col = "patch_id"
label_col = "label"
num_classes = 10
clm = CassandraListManager(ap, [cassandra_ip])
clm.set_config(
    table="imagenette.ids_224",
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)
clm.read_rows_from_db()
clm.split_setup(split_ratios=[7, 2, 1])
cd = CassandraDataset(ap, [cassandra_ip])
cd.use_splits(clm)
cd.set_config(
    table="imagenette.data_224",
    bs=28,
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)

for _ in trange(5):
    cd.rewind_splits(shuffle=True)
    for i in range(cd.num_batches[0]):
        x, y = cd.load_batch()


# RGB with augmentations
from torchvision import transforms
augs_fn = "/tmp/augs.pt"
# rescale and normalize
n_scale = 255.  # divide by 255
n_mean = n_scale*np.array((0.485, 0.456, 0.406)).tolist()
n_std = n_scale*np.array((0.229, 0.224, 0.225)).tolist()
augs = torch.nn.Sequential(
    transforms.Normalize(n_mean, n_std, inplace=True),
)
s_augs = torch.jit.script(augs)
s_augs.save(augs_fn)

cd.set_config(rgb=True)
for _ in trange(5):
    cd.rewind_splits(shuffle=True)
    for i in range(cd.num_batches[0]):
        x, y = cd.load_batch()