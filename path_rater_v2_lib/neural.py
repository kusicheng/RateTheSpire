# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *

def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but the installed PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled PyTorch wheel or use --device cpu."
        )
    return device


class PathRaterNet(nn.Module):
    """A compact sequence model for visible path plus current run state."""

    def __init__(
        self,
        numeric_dim: int,
        cat_cardinalities: list[int],
        d_model: int = 64,
        use_position: bool = True,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        self.use_position = use_position
        self.use_attention_pooling = use_attention_pooling
        self.room_embedding = nn.Embedding(len(ROOM_TYPES), d_model, padding_idx=ROOM_TO_ID["PAD"])
        if use_position:
            self.position_embedding = nn.Embedding(MAX_PATH_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.path_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, 8) for cardinality in cat_cardinalities]
        )
        if use_attention_pooling:
            self.path_attention = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        combined_dim = d_model + numeric_dim + 8 * len(cat_cardinalities)
        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.floor_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        rooms: torch.Tensor,
        cats: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        room_emb = self.room_embedding(rooms)
        if self.use_position:
            positions = torch.arange(rooms.size(1), device=rooms.device).unsqueeze(0).expand_as(rooms)
            room_emb = room_emb + self.position_embedding(positions)
        padding_mask = rooms.eq(ROOM_TO_ID["PAD"])
        encoded = self.path_encoder(room_emb, src_key_padding_mask=padding_mask)
        if self.use_attention_pooling:
            attn_scores = self.path_attention(encoded).squeeze(-1).masked_fill(padding_mask, -1e9)
            attn = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            path_vec = (encoded * attn).sum(dim=1)
        else:
            mask = (~padding_mask).float().unsqueeze(-1)
            path_vec = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        cat_vec = torch.cat(
            [embedding(cats[:, idx]) for idx, embedding in enumerate(self.cat_embeddings)],
            dim=1,
        )
        features = torch.cat([numeric, path_vec, cat_vec], dim=1)
        win_logits = self.head(features).squeeze(1)
        if return_aux:
            return win_logits, torch.sigmoid(self.floor_head(features).squeeze(1))
        return win_logits
