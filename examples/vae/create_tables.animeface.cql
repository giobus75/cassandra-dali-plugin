//////////////////////////////////////////////////////////////////////
// imagenet: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

CREATE KEYSPACE IF NOT EXISTS animeface WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS animeface.data_orig(
  patch_id uuid,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS animeface.metadata_orig(
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
