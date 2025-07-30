from loader.MSVD import MSVD
from config import TrainConfig as C
from models.abd_transformer import ABDTransformer
import torch
from utils import dict_to_cls


# Load checkpoint and config
checkpoint = torch.load("checkpoints/best.ckpt", map_location="cpu")
config = dict_to_cls(checkpoint['config'])
corpus = MSVD(config)


# Build Model
vocab = corpus.vocab
""" Build Models """
try:
    model = ABDTransformer(vocab, config.feat.size, config.transformer.d_model, config.transformer.d_ff,
                           config.transformer.n_heads, config.transformer.n_layers, config.transformer.dropout,
                           config.feat.feature_mode, n_heads_big=config.transformer.n_heads_big,
                           select_num=config.transformer.select_num)
except:
    model = ABDTransformer(vocab, config.feat.size, config.transformer.d_model, config.transformer.d_ff,
                           config.transformer.n_heads, config.transformer.n_layers, config.transformer.dropout,
                           config.feat.feature_mode, n_heads_big=config.transformer.n_heads_big)
model.load_state_dict(checkpoint['abd_transformer'])


# Move model to cpu
model.device = "cpu"
model = model.to("cpu")

# Load extracted features
image_feats = torch.load('features/image_feats.pt', map_location="cpu")
motion_feats = torch.load('features/motion_feats.pt', map_location="cpu")
obect_feats = torch.load('features/object_feats.pt', map_location="cpu")
rel_feats = torch.load('features/rel_feats.pt', map_location="cpu")

# Load config parameters
beam_size = config.beam_size
max_len = config.loader.max_caption_len
feature_mode = config.feat.feature_mode
feats = (image_feats, motion_feats, obect_feats, rel_feats)

# Inference with beam search
model.eval()
with torch.no_grad():
    r2l_captions, l2r_captions = model.beam_search_decode(
        feats, beam_size, max_len)
    # r2l_captions = [idxs_to_sentence(caption, vocab.idx2word, BOS_idx) for caption in r2l_captions]
    l2r_captions = [" ".join(caption[0].value) for caption in l2r_captions]
    r2l_captions = [" ".join(caption[0].value) for caption in r2l_captions]

    print(f"Left to Right Captions: {l2r_captions}")
