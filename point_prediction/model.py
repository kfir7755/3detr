import numpy as np

# Monkey-patch np.int to work like the built-in int (restoring old behavior)
np.int = int

import torch
import torch.nn as nn
from models.model_3detr import build_encoder, build_decoder
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine


class Landmark3DETR(nn.Module):
    def __init__(self, pre_encoder, encoder, decoder, num_landmarks=17, embed_dim=256):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.decoder = decoder

        # Learnable embeddings for the 17 distinct landmarks
        # These replace the FPS-sampled queries from the original 3DETR
        self.query_embed = nn.Embedding(num_landmarks, embed_dim)

        # Positional embedding for the point cloud points (fourier/sine)
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=embed_dim, pos_type="fourier", normalize=True
        )

        # Projection layer to align encoder outputs with decoder dimension
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=embed_dim,
            hidden_dims=[embed_dim],
            output_dim=embed_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        # Regression Head: Features -> XYZ
        # Predicts the offset or absolute coordinate for each landmark
        self.landmark_head = GenericMLP(
            input_dim=embed_dim,
            hidden_dims=[embed_dim, 128, 64],
            output_dim=3,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=False,
            output_use_norm=False,
            output_use_bias=True
        )

    def forward(self, inputs):
        """
        Args:
            inputs: dict containing 'point_clouds' of shape (B, N, 11)
                    [x, y, z, nx, ny, nz, seg1, seg2, seg3, seg4, seg5]
        Returns:
            coords: (B, 17, 3) Predicted landmark coordinates
        """
        # 1. Parse Inputs
        # point_clouds: (B, N, 11)
        pc = inputs["point_clouds"]
        xyz = pc[..., :3].contiguous()  # (B, N, 3)
        features = pc[..., 3:].transpose(1, 2).contiguous()  # (B, 8, N) - Normals + OneHot

        # 2. Pre-Encoder (PointNet++ layer)
        # enc_xyz: (B, N, 3), enc_features: (B, C, N)
        # Note: We keep N=2048 (no downsampling) based on your description
        enc_xyz, enc_features, _ = self.pre_encoder(xyz, features)

        # 3. Transformer Encoder
        # Permute to (N, B, C) for Transformer
        enc_features = enc_features.permute(2, 0, 1)

        # The encoder returns (xyz, output, inds)
        # We only need the output features here
        _, enc_features_trans, _ = self.encoder(enc_features, xyz=enc_xyz)

        # Project encoder features (B, C, N) -> (B, C, N)
        enc_features_proj = self.encoder_to_decoder_projection(
            enc_features_trans.permute(1, 2, 0)
        )

        # 4. Prepare Embeddings
        # Calculate positional embeddings for the points based on XYZ
        # We compute min/max per batch for normalization
        point_cloud_dims = [
            xyz.min(dim=1)[0],
            xyz.max(dim=1)[0]
        ]
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # Prepare Decoder Inputs
        # Memory (Encoder Output): (N, B, C)
        memory = enc_features_proj.permute(2, 0, 1)
        # Positional Embeddings for Memory: (N, B, C)
        pos = enc_pos.permute(2, 0, 1)

        # Queries: (17, B, C)
        batch_size = xyz.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = torch.zeros_like(query_pos)

        # 5. Transformer Decoder
        # The decoder returns a tuple: (intermediate_stack, attentions)
        # [0] selects the stack of intermediate outputs: shape (num_layers, num_queries, batch, embed_dim)
        # [-1] selects the last layer's output: shape (num_queries, batch, embed_dim)
        decoder_output = self.decoder(
            tgt, memory, query_pos=query_pos, pos=pos
        )[0][-1]  # <--- Added [-1] here

        # 6. Regression Head
        # Input: (B, C, 17)
        decoder_output = decoder_output.permute(1, 2, 0)
        coords = self.landmark_head(decoder_output)  # (B, 3, 17)

        return coords.transpose(1, 2)  # Return (B, 17, 3)


def build_landmark_3detr(args):
    # 1. Build Pre-encoder adapted for 8 feature channels
    # (3 normals + 5 segment labels = 8)
    # We maintain 2048 points (npoint=2048)
    pre_encoder = PointnetSAModuleVotes(
        radius=0.1,
        nsample=64,
        npoint=6000,
        mlp=[8, 64, 128, args.enc_dim],  # First dim is 8 for your features
        normalize_xyz=False,
    )

    # 2. Build Encoder (Standard 3DETR Vanilla or Masked)
    encoder = build_encoder(args)

    # 3. Build Decoder (Standard 3DETR Decoder)
    decoder = build_decoder(args)

    # 4. Assemble Model
    model = Landmark3DETR(
        pre_encoder=pre_encoder,
        encoder=encoder,
        decoder=decoder,
        num_landmarks=17,
        embed_dim=args.enc_dim
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)
    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )
    args = parser.parse_args()
    model = build_landmark_3detr(args).to('cuda')
    # print(model)
    dummy_inputs = {
        "point_clouds": torch.randn(2, 2048, 11).to('cuda')  # (B, N, 11)
    }
    outputs = model(dummy_inputs)
    print("Output landmark coordinates shape:", outputs.shape)  # Expected: (2, 17, 3)
