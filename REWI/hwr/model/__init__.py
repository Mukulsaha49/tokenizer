from torch.nn import Sequential
from .conv import BLCNN
from .lstm import LSTM


def get_model(
    arch_en, arch_de, in_chan, num_cls,
    ratio_ds=8, len_seq=0, **kwargs  # for extra cfg keys
):
    # === Encoder selection ===
    if arch_en == 'blcnn':
        encoder = BLCNN(in_chan)
    else:
        raise ValueError(f"Unknown encoder type: {arch_en}")

    # === Decoder selection ===
    if arch_de == 'lstm':
        decoder = LSTM(encoder.size_out, num_cls)
    else:
        raise ValueError(f"Unknown decoder type: {arch_de}")

    return Sequential(encoder, decoder)
