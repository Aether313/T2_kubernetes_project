#Edited by Feiyang
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "imdb_rating_cleaned.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("Data preview:")
print(df.head())

# Handle Boxoffice
df['Boxoffice'] = pd.to_numeric(df['Boxoffice'].replace('Unknown', np.nan), errors='coerce')
if df['Boxoffice'].isna().sum() > 0:
    df['Boxoffice'] = df['Boxoffice'].fillna(df['Boxoffice'].median())

# Helper functions
def take_first(x):
    if pd.isna(x):
        return "Unknown"
    parts = [p.strip() for p in str(x).split(',') if p.strip() != '']
    return parts[0] if parts else "Unknown"

def split_multi(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split(',') if p.strip() != '']

# Process primary categorical fields
df['Director_primary'] = df['Director'].apply(take_first)
df['Writer_primary'] = df['Writer'].apply(take_first)
df['Prod_primary']   = df['Production House'].apply(take_first)

# Process multi-valued categorical fields
multi_cols = ['Genre', 'Languages', 'Tags', 'Country Availability']
for c in multi_cols:
    df[c + "_list"] = df[c].apply(split_multi)

# Numerical & target
num_cols = ['IMDb Votes', 'Runtime', 'Hidden Gem Score', 'Boxoffice']
target_col = 'IMDb Score'

# Vocabulary builders
def build_vocab(series, min_freq=1):
    counts = series.value_counts()
    vocab = ['[PAD]', '[UNK]'] + counts[counts >= min_freq].index.tolist()
    return {v: i for i, v in enumerate(vocab)}

def build_token_map(list_series, min_freq=1):
    from collections import Counter
    c = Counter()
    for lst in list_series:
        c.update(lst)
    tokens = ['[PAD]', '[UNK]'] + [tok for tok, ct in c.items() if ct >= min_freq]
    return {v: i for i, v in enumerate(tokens)}

director_map = build_vocab(df['Director_primary'])
writer_map   = build_vocab(df['Writer_primary'])
prod_map     = build_vocab(df['Prod_primary'])

genre_map   = build_token_map(df['Genre_list'])
lang_map    = build_token_map(df['Languages_list'])
tags_map    = build_token_map(df['Tags_list'])
country_map = build_token_map(df['Country Availability_list'])

# Encoding helpers
def lists_to_sequences(list_series, token_map, max_len):
    unk = token_map.get('[UNK]', 1)
    pad = token_map.get('[PAD]', 0)
    seqs = [[token_map.get(t, unk) for t in lst][:max_len] for lst in list_series]
    return pad_sequences(seqs, padding='post', truncating='post', value=pad, maxlen=max_len)

def series_to_ids(series, mapping):
    unk = mapping.get('[UNK]', 1)
    return series.map(lambda x: mapping.get(x, unk)).astype(np.int32).values

# Numerical input
X_num = df[num_cols].fillna(0).values.astype(np.float32)
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Single-value categorical
X_director = series_to_ids(df['Director_primary'], director_map)
X_writer   = series_to_ids(df['Writer_primary'], writer_map)
X_prod     = series_to_ids(df['Prod_primary'], prod_map)

# Multi-value categorical
X_genre_seq   = lists_to_sequences(df['Genre_list'], genre_map, max_len=6)
X_lang_seq    = lists_to_sequences(df['Languages_list'], lang_map, max_len=4)
X_tags_seq    = lists_to_sequences(df['Tags_list'], tags_map, max_len=12)
X_country_seq = lists_to_sequences(df['Country Availability_list'], country_map, max_len=10)

y = df[target_col].values.astype(np.float32)

# Train/val/test split
X_train_ids, X_test_ids, y_train, y_test = train_test_split(
    np.arange(len(df)), y, test_size=0.15, random_state=42)
X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    X_train_ids, y_train, test_size=0.1765, random_state=42)

def make_input_dict(indices):
    return {
        'num_input': X_num[indices],
        'director_input': X_director[indices],
        'writer_input': X_writer[indices],
        'prod_input': X_prod[indices],
        'genre_seq': X_genre_seq[indices],
        'lang_seq': X_lang_seq[indices],
        'tags_seq': X_tags_seq[indices],
        'country_seq': X_country_seq[indices],
    }

train_inputs = make_input_dict(X_train_ids)
val_inputs   = make_input_dict(X_val_ids)
test_inputs  = make_input_dict(X_test_ids)

# Model building
def emb_size(n_unique):
    return int(min(50, max(4, n_unique // 2)))

num_input = layers.Input(shape=(len(num_cols),), name='num_input')
director_input = layers.Input(shape=(), dtype='int32', name='director_input')
writer_input   = layers.Input(shape=(), dtype='int32', name='writer_input')
prod_input     = layers.Input(shape=(), dtype='int32', name='prod_input')
genre_seq      = layers.Input(shape=(X_genre_seq.shape[1],), dtype='int32', name='genre_seq')
lang_seq       = layers.Input(shape=(X_lang_seq.shape[1],), dtype='int32', name='lang_seq')
tags_seq       = layers.Input(shape=(X_tags_seq.shape[1],), dtype='int32', name='tags_seq')
country_seq    = layers.Input(shape=(X_country_seq.shape[1],), dtype='int32', name='country_seq')

def embed_single(inp, vocab_map, name):
    emb = layers.Embedding(input_dim=len(vocab_map), output_dim=emb_size(len(vocab_map)), name=name)(inp)
    return layers.Reshape((emb.shape[-1],))(emb)

def embed_multi(inp, vocab_map, name):
    emb = layers.Embedding(input_dim=len(vocab_map), output_dim=emb_size(len(vocab_map)),
                            mask_zero=True, name=name)(inp)
    return layers.GlobalAveragePooling1D()(emb)

director_emb = embed_single(director_input, director_map, 'director_emb')
writer_emb   = embed_single(writer_input, writer_map, 'writer_emb')
prod_emb     = embed_single(prod_input, prod_map, 'prod_emb')
genre_emb    = embed_multi(genre_seq, genre_map, 'genre_emb')
lang_emb     = embed_multi(lang_seq, lang_map, 'lang_emb')
tags_emb     = embed_multi(tags_seq, tags_map, 'tags_emb')
country_emb  = embed_multi(country_seq, country_map, 'country_emb')

conc = layers.Concatenate()([num_input,
                             director_emb, writer_emb, prod_emb,
                             genre_emb, lang_emb, tags_emb, country_emb])
x = layers.Dense(256, activation='relu')(conc)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1, activation='linear')(x)

model = Model(inputs=[num_input, director_input, writer_input, prod_input,
                      genre_seq, lang_seq, tags_seq, country_seq],
              outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])

BATCH_SIZE = 64
def make_dataset(inputs_dict, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((inputs_dict, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_inputs, y_train, shuffle=True)
val_ds   = make_dataset(val_inputs, y_val)
test_ds  = make_dataset(test_inputs, y_test)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
                                        save_best_only=True, monitor='val_loss')
]

history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

print("\nTest Evaluation:")
model.evaluate(test_ds)

# Save model for inference
model.save(os.path.join(MODEL_DIR, "model.h5"), include_optimizer=False)
print(f"Model saved to {os.path.join(MODEL_DIR, 'model.h5')}")

# Save preprocessing artifacts
preproc = {
    'director_map': director_map,
    'writer_map': writer_map,
    'prod_map': prod_map,
    'genre_map': genre_map,
    'lang_map': lang_map,
    'tags_map': tags_map,
    'country_map': country_map,
    'num_cols': num_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_var': scaler.var_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}
with open(os.path.join(MODEL_DIR, 'preproc.pkl'), 'wb') as f:
    pickle.dump(preproc, f)

print(f"Preprocessing artifacts saved to {os.path.join(MODEL_DIR, 'preproc.pkl')}")
