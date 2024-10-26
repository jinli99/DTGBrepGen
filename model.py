import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder
from diffusers.models.unets.unet_1d_blocks import ResConvBlock, SelfAttention1d, get_down_block, get_up_block, Upsample1d
from diffusers.models.attention_processor import SpatialNorm

import math
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as ff
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from utils import xe_mask, assert_correctly_masked, masked_softmax, pad_and_stack
from topology.transfer import edge_vert_trans

from utils import make_mask


class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class Encoder1D(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock1D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv1d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv1d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )

                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)[0]

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder1D(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpBlock1D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds, use_reentrant=False
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, use_reentrant=False
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            # sample = sample.to(upscale_dtype)
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class AutoencoderKLFastEncode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, str, str, str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, str, str, str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, int, int, int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

    def forward(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        latent_z = DiagonalGaussianDistribution(moments).mode()  # mode converge faster
        return latent_z


class AutoencoderKLFastDecode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, str, str, str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, str, str, str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, int, int, int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        decoded = self._decode(z).sample
        return decoded


class AutoencoderKL1D(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, str, str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str, str, str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int, int, int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            scaling_factor: float = 0.18215,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv1d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.use_tiling = use_tiling

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.enable_tiling(False)

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
            self,
            sample: torch.FloatTensor,
            sample_posterior: bool = False,  # True
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


class AutoencoderKL1DFastEncode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, str, str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str, str, str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, int, int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder1D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv = nn.Conv1d(2 * latent_channels, 2 * latent_channels, 1)

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,  # True
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        latent_z = posterior.mode()
        return latent_z


class AutoencoderKL1DFastDecode(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, str, str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str, str, str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int, int, int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 32,
            scaling_factor: float = 0.18215,
    ):
        super().__init__()

        # pass init params to Decoder
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.post_quant_conv = nn.Conv1d(latent_channels, latent_channels, 1)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        dec = self._decode(z).sample
        return dec


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probability. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = ff.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class FaceGeomTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probability. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, n_head: int, dim_ffX: int = 512,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = FaceGeomNodeEdgeBlock(dx, de, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.activation = ff.relu

    def forward(self, X: Tensor, e_add, e_mul, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX = self.self_attn(X, e_add, e_mul, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        return X


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n_head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class FaceGeomNodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # Output layers
        self.x_out = Linear(dx, dx)

    def forward(self, X, e_add, e_mul, node_mask):

        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n_head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = e_mul * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((bs, n, n, self.n_head, self.df))

        E2 = e_add * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((bs, n, n, self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        newX = weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        return newX


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['x']
        self.out_dim_E = output_dims['e']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['x'], hidden_mlp_dims['x']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['x'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['e'], hidden_mlp_dims['e']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['e'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['x'], output_dims['x']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['e']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['e'], output_dims['e']))

        # self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, x, e, y, node_mask):
        bs, n = x.shape[0], x.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(e).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = x[..., :self.out_dim_X]
        E_to_out = e[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(e)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        x, e = xe_mask(x=self.mlp_in_X(x), e=new_E, node_mask=node_mask)
        # after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            x, e, y = layer(x, e, y, node_mask)

        x = self.mlp_out_X(x)
        e = self.mlp_out_E(e)

        x = (x + X_to_out)
        e = (e + E_to_out) * diag_mask

        # if self.out_dim_y > 0:
        #     y = self.mlp_out_y(y)
        #     y = y + y_to_out
        # else:
        #     y = y_to_out

        e = 1/2 * (e + torch.transpose(e, 1, 2))

        x, e = xe_mask(x, e, node_mask=node_mask)

        return x, e, y

        # return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in")

    def forward(self, x):
        return self.embed(x)


class FaceBboxTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict, edge_classes: int,
                 act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), use_cf=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_cf = use_cf

        self.mlp_in_X = nn.Sequential(nn.Linear(6, hidden_mlp_dims['x']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['x'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Embedding(edge_classes, hidden_dims['de'])

        self.embed_dim = hidden_dims['dx']

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
            nn.LayerNorm(hidden_dims['dx']),
            nn.SiLU(),
            nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
        )

        self.tf_layers = nn.ModuleList([FaceGeomTransformerLayer(dx=hidden_dims['dx'],
                                                                 de=hidden_dims['de'],
                                                                 n_head=hidden_dims['n_head'],
                                                                 dim_ffX=hidden_dims['dim_ffX'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['x'], 6))

        # FiLM E to X
        self.e_add = Linear(hidden_dims['de'], hidden_dims['dx'])
        self.e_mul = Linear(hidden_dims['de'], hidden_dims['dx'])

        if self.use_cf:
            self.class_embed = Embedder(11, self.embed_dim)

    def forward(self, x_t, e, mask, class_label, t):
        """
        Args:
        - x_t: b*nf*6
        - e: b*nf*nf
        - mask: b*nf
        - class_label: b*1
        - t: b*1
        """

        X_to_out = x_t

        new_E = self.mlp_in_E(e)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        time_embeds = self.time_embed(sincos_embedding(t, self.embed_dim))  # b*1*embed_dim

        tokens = self.mlp_in_X(x_t)+time_embeds
        if self.use_cf:
            if self.training:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(x_t.shape[0], 1) <= 0.1
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label)
            tokens += c_embeds

        x, e = xe_mask(x=tokens, e=new_E, node_mask=mask)

        e_add = self.e_add(e)
        e_mul = self.e_mul(e)
        for layer in self.tf_layers:
            x = layer(x, e_add, e_mul, mask)

        x = self.mlp_out_X(x)
        x = (x + X_to_out)

        x, _ = xe_mask(x=x, node_mask=mask)

        return x


class VertGeomTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), use_cf=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_cf = use_cf

        self.mlp_in_X = nn.Sequential(nn.Linear(3, hidden_mlp_dims['x']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['x'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Embedding(2, hidden_dims['de'])

        self.embed_dim = hidden_dims['dx']

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
            nn.LayerNorm(hidden_dims['dx']),
            nn.SiLU(),
            nn.Linear(hidden_dims['dx'], hidden_dims['dx']),
        )

        self.face_bbox_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.tf_layers = nn.ModuleList([FaceGeomTransformerLayer(dx=hidden_dims['dx'],
                                                                 de=hidden_dims['de'],
                                                                 n_head=hidden_dims['n_head'],
                                                                 dim_ffX=hidden_dims['dim_ffX'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['x'], 3))

        # FiLM E to X
        self.e_add = Linear(hidden_dims['de'], hidden_dims['dx'])
        self.e_mul = Linear(hidden_dims['de'], hidden_dims['dx'])

        if self.use_cf:
            self.class_embed = Embedder(11, self.embed_dim)

    def forward(self, x_t, e, mask, vertFace_bbox, vertFace_mask, class_label, t):
        """
        Args:
        - x_t: b*nv*3
        - e: b*nv*nv*2
        - mask: b*nv
        - vertFace_bbox: b*nv*vf*6
        - vertFace_mask: b*nv*vf
        - class_label: b*1
        - t: b*1
        """

        X_to_out = x_t

        new_E = self.mlp_in_E(e)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        time_embeds = self.time_embed(sincos_embedding(t, self.embed_dim))  # b*1*embed_dim

        vertFace_embeds = self.face_bbox_embed(vertFace_bbox)               # b*nv*vf*embed_dim
        vertFace_embeds = vertFace_embeds.sum(-2) / (vertFace_mask.sum(-1, keepdim=True)+1e-8)     # b*nv*embed_dim

        tokens = self.mlp_in_X(x_t)+time_embeds+vertFace_embeds
        if self.use_cf:
            if self.training:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(x_t.shape[0], 1) <= 0.1
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label)
            tokens += c_embeds

        x, e = xe_mask(x=tokens, e=new_E, node_mask=mask)

        e_add = self.e_add(e)
        e_mul = self.e_mul(e)
        for layer in self.tf_layers:
            x = layer(x, e_add, e_mul, mask)

        x = self.mlp_out_X(x)
        x = (x + X_to_out)

        x, _ = xe_mask(x=x, node_mask=mask)

        return x


def sincos_embedding(inputs, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param inputs: an N-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freq = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=inputs.device)
    for _ in range(len(inputs.size())):
        freq = freq[None]
    args = inputs.unsqueeze(-1).float() * freq
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FaceFeatExtract(nn.Module):
    def __init__(self, max_edge, edge_classes, int_face_features=8, out_face_features=16):
        super(FaceFeatExtract, self).__init__()
        self.feat_extractor = GraphFeatModel(input_face_features=int_face_features,
                                             num_face_features=out_face_features)
        self.face_nEdge_embedding = nn.Embedding(max_edge, int_face_features)
        self.edge_classes = edge_classes
        self.max_edge = max_edge

    @staticmethod
    def get_edge_index(edgeFace_adj, edge_mask):
        batch_size, ne, _ = edgeFace_adj.shape

        face_offsets = torch.cumsum(torch.max(edgeFace_adj.reshape(batch_size, -1), dim=1)[0] + 1, dim=0)
        face_offsets = torch.cat([torch.tensor([0], device=edgeFace_adj.device), face_offsets[:-1]])

        edgeFace_adj_offset = edgeFace_adj + face_offsets.reshape(-1, 1, 1)

        edge_index = edgeFace_adj_offset[edge_mask]

        return edge_index  # Ne*2

    def forward(self, edgeFace_adj, edge_mask, edgeVert_adj=None):
        """
        Args:
            edgeFace_adj: A tensor of shape [batch_size, ne, 2].
            edge_mask: A tensor of shape [batch_size, ne].
            edgeVert_adj: A tensor of shape [batch_size, ne, 2].
            """

        # compute face feature
        edge_index = self.get_edge_index(edgeFace_adj, edge_mask)  # Ne*2
        assert (edge_index[:, 0] - edge_index[:, 1]).abs().min() > 0
        face_nEdges = torch.bincount(edge_index.flatten(), minlength=edge_index.max().item() + 1)
        edge_index_unique, edge_index_num = torch.unique(
            torch.sort(edge_index, dim=1).values, dim=0, return_counts=True)
        assert face_nEdges.max() <= self.max_edge
        face_embed = self.face_nEdge_embedding(face_nEdges)             # nf*input_face_features
        assert edge_index_num.max() < self.edge_classes
        face_feat = self.feat_extractor(face_embed, edge_index_unique.transpose(0, 1),
                                        edge_index_num / self.edge_classes)

        # gather edges to batch
        batch_size = edgeFace_adj.shape[0]
        edge_num_per = edge_mask.sum(1)  # b
        if edgeVert_adj is None:
            # assign edge feature

            edge_feat = face_feat[edge_index]  # Ne*2*embed_features

            edge_embed = torch.zeros((batch_size, edge_num_per.max().item(), edge_feat.size(-1)),
                                     dtype=edge_feat.dtype, device=edge_feat.device)  # b*ne*m
            start_idx = 0
            for i in range(batch_size):
                num_edges = edge_num_per[i].item()
                end_idx = start_idx + num_edges

                edge_embed[i, :num_edges, :] = edge_feat[start_idx:end_idx, ...].mean(1)

                start_idx = end_idx
            return edge_embed
        else:
            face_num_per = torch.max(edgeFace_adj.flatten(1, 2), dim=1)[0] + 1  # b
            start_idx = 0
            vertFace_embed = []
            for i in range(batch_size):
                num_edges = edge_num_per[i].item()
                vertEdge_adj = edge_vert_trans(edgeVert_adj=edgeVert_adj[i, :num_edges])
                vertFace_adj = [torch.unique(edgeFace_adj[i][j].reshape(-1)) for j in vertEdge_adj]    # List[Tensor: vf]
                vertFace_embed.append(torch.stack([face_feat[j+start_idx].mean(0) for j in vertFace_adj]))       # nv*m
                start_idx += face_num_per[i]
            vertFace_embed, _ = pad_and_stack(vertFace_embed)                                          # b*nv*m
            return vertFace_embed


class EdgeGeomTransformer(nn.Module):
    """
    Transformer-based latent diffusion model for edge feature
    """

    def __init__(self, n_layers: int, edge_geom_dim: int, d_model: int, nhead: int, use_cf: bool):
        super(EdgeGeomTransformer, self).__init__()
        self.embed_dim = d_model
        self.use_cf = use_cf

        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, norm_first=True,
                                           dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(self.embed_dim))

        self.edge_geom_embed = nn.Sequential(
            nn.Linear(edge_geom_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.face_bbox_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.edge_vert_embed = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, edge_geom_dim),
        )

        if self.use_cf:
            self.class_embed = Embedder(11, self.embed_dim)

        return

    def forward(self, x_t, edgeFace_bbox, edgeVert_geom, edge_mask, class_label, t):   # b*ne*12, b*ne*2*6, b*ne*2*3, b*ne, b*ne*2, b*1, b*1
        """ forward pass """

        time_embeds = self.time_embed(sincos_embedding(t, self.embed_dim))    # b*1*embed_dim
        face_bbox_embeds = self.face_bbox_embed(edgeFace_bbox)                # b*ne*2*embed_dim

        edge_vert_embeds = self.edge_vert_embed(edgeVert_geom).mean(-2)       # b*ne*embed_dim
        edge_geom_embeds = self.edge_geom_embed(x_t)                          # b*ne*embed_dim

        tokens = edge_geom_embeds + edge_vert_embeds + time_embeds + face_bbox_embeds.mean(-2)   # b*ne*embed_dim

        if self.use_cf:
            if self.training:
                # randomly set 10% to uncond label
                uncond_mask = torch.rand(x_t.shape[0], 1) <= 0.1
                class_label[uncond_mask] = 0
            c_embeds = self.class_embed(class_label)
            tokens += c_embeds

        output = self.net(
            src=tokens.permute(1, 0, 2),
            src_key_padding_mask=~edge_mask,
        ).transpose(0, 1)

        pred = self.fc_out(output)     # b*ne*12
        return pred


class FaceGeomTransformer(nn.Module):
    """
    Transformer-based latent diffusion model for face feature
    """

    def __init__(self, n_layers: int, face_geom_dim: int):
        super().__init__()
        self.embed_dim = 768
        assert not self.embed_dim % 16

        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, norm_first=True,
                                           dim_feedforward=1024, dropout=0.1)
        self.net = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(self.embed_dim))

        self.face_geom_embed = nn.Sequential(
            nn.Linear(face_geom_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.face_bbox_embed = nn.Sequential(
            nn.Linear(6, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.faceVert_embed = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.faceEdge_embed = nn.Sequential(
            nn.Linear(12, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, face_geom_dim),
        )

    def forward(self, x_t, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask, faceEdge_mask, t):
        """
            Args:
                x_t: [batch_size, nf, 48]
                face_bbox: [batch_size, nf, 6]
                faceVert_geom: [batch_size, nf, fv, 3]
                faceEdge_geom: [batch_size, nf, fe, 12]
                face_mask: [batch_size, nf]
                faceVert_mask: [batch_size, nf, fv]
                faceEdge_mask: [batch_size, nf, fe]
                t: [batch_size, 1]
            Returns:
                Noise prediction with shape [batch_size, nf, 48]
        """

        time_embeds = self.time_embed(sincos_embedding(t, self.embed_dim))    # b*1*embed_dim
        face_bbox_embeds = self.face_bbox_embed(face_bbox)                    # b*nf*embed_dim
        # face_bbox_embeds = 0
        face_geom_embeds = self.face_geom_embed(x_t)                          # b*nf*embed_dim

        faceVert_embeds = self.faceVert_embed(faceVert_geom)                  # b*nf*fv*embed_dim
        faceVert_embeds = faceVert_embeds.sum(-2) / (faceVert_mask.sum(-1, keepdim=True)+1e-8)     # b*nf*embed_dim

        faceEdge_embeds = self.faceEdge_embed(faceEdge_geom)                  # b*nf*fv*embed_dim
        faceEdge_embeds = faceEdge_embeds.sum(-2) / (faceEdge_mask.sum(-1, keepdim=True)+1e-8)     # b*nf*embed_dim

        tokens = face_geom_embeds + face_bbox_embeds + faceVert_embeds + faceEdge_embeds + time_embeds         # b*nf*embed_dim

        output = self.net(
            src=tokens.permute(1, 0, 2),
            src_key_padding_mask=~face_mask,
        ).transpose(0, 1)

        pred = self.fc_out(output)     # b*nf*48
        return pred


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=16, num_heads=4, num_layers=4, output_dim=1):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(GATConv(num_features, hidden_dim, heads=num_heads, concat=True, dropout=0.6))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=0.6))
        self.layers.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.6))

        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers - 1)])

        self.output_layer = torch.nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index)
            x = torch.nn.functional.elu(x)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
        x = self.layers[-1](x, edge_index)
        return self.output_layer(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FaceEdgeModel(nn.Module):
    def __init__(self, nf=32, d_model=128, nhead=4, dim_feedforward=1024, n_layers=4, num_categories=5, use_cf=False):
        super().__init__()
        self.nf = nf
        self.seq_len = int(nf * (nf - 1) / 2)
        self.d_model = d_model
        self.use_cf = use_cf
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_categories = num_categories
        self.GATmodel = GAT(num_features=1).to(device)
        self.GAToptimizer = torch.optim.Adam(self.GATmodel.parameters(), lr=0.01)
        self.mlp = MLP(input_dim=self.seq_len * 2, hidden_dim=256, output_dim=self.seq_len * self.d_model).to(device)

        self.embedding = nn.Embedding(num_categories + 1, d_model)
        self.positional_encoding = self.create_positional_encoding()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_layer = nn.Linear(d_model, num_categories)

        if self.use_cf:
            self.class_embed = Embedder(11, self.d_model)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_positional_encoding(self):
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def create_positional_encode_fef(self, src):
        batch_size, seq_len = src.size()
        n = int((1 + (1 + 8 * seq_len) ** 0.5) / 2)
        device = src.device

        upper_indices = torch.triu_indices(n, n, offset=1, device=device)
        encoded_positions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        mask = src > 0
        idx = mask.nonzero(as_tuple=False)

        if idx.numel() > 0:
            batch_indices = idx[:, 0]
            seq_indices = idx[:, 1]
            i_indices = upper_indices[0][seq_indices]
            j_indices = upper_indices[1][seq_indices]

            position_codes = i_indices * 10 + j_indices
            encoded_positions[batch_indices, seq_indices] = position_codes
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        position = encoded_positions.float().unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model)
        )

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe

    def create_GAT_encode(self, src):
        batch_size, seq_len = src.size()
        n = int((1 + (1 + 8 * seq_len) ** 0.5) / 2)
        device = src.device

        upper_indices = torch.triu_indices(n, n, offset=1).to(device)

        data_list = []

        for i in range(batch_size):
            src_i = src[i]
            edge_mask = src_i > 0

            expanded_edge_indices = upper_indices[:, edge_mask].repeat_interleave(src_i[edge_mask].to(torch.long),
                                                                                  dim=1)

            reverse_edge_indices = torch.stack([expanded_edge_indices[1], expanded_edge_indices[0]], dim=0)
            full_edge_index = torch.cat([expanded_edge_indices, reverse_edge_indices], dim=1).to(torch.long)

            x = torch.ones(n, 1).to(device)

            data = Data(x=x, edge_index=full_edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list).to(device)
        out = self.GATmodel(batch)

        node_counts = torch.bincount(batch.batch)
        node_features_list = torch.split(out, tuple(node_counts.cpu().numpy()))

        node_features = torch.stack(node_features_list, dim=0)

        return node_features.squeeze(-1)

    def compute_GAT_encode(self, node_features):
        batch_size, n = node_features.shape
        device = node_features.device

        upper_indices = torch.triu_indices(n, n, offset=1).to(device)
        seq_len = upper_indices.size(1)

        node_i_features = node_features[:, upper_indices[0]]
        node_j_features = node_features[:, upper_indices[1]]

        edge_features = torch.cat([node_i_features, node_j_features], dim=-1)

        edge_features = self.mlp(edge_features)
        edge_features = edge_features.view(batch_size, seq_len, self.d_model)
        layer_norm = torch.nn.LayerNorm(self.d_model).to(edge_features.device)
        edge_features = layer_norm(edge_features)
        return edge_features

    def encode(self, src, class_label):
        """
        Args:
            src: [batch_size, seq_len]
            class_label: [batch_size, 1]
        """

        fef_pos_encoding = self.create_positional_encode_fef(src).to(src.device)
        # node_encode = self.create_GAT_encode(src)
        # GAT_encode = self.compute_GAT_encode(node_encode)
        src = self.embedding(src) + self.positional_encoding #+ GAT_encode

        if self.use_cf:
            src += self.class_embed(class_label)

        # src = self.embedding(src) + self.positional_encoding
        # src shape after embedding: [batch_size, seq_len, d_model]
        memory = self.transformer_encoder(src.transpose(0, 1)).transpose(0, 1)     # [batch_size, seq_len, d_model]
        mu = self.fc_mu(memory.mean(dim=1))                                                   # [batch_size, d_model]
        logvar = self.fc_logvar(memory.mean(dim=1))                                           # [batch_size, d_model]
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, tgt, class_label):
        """
        Args:
            z: [batch_size, d_model]
            tgt: [batch_size, current_seq_len]
            class_label: [batch_size, 1]
        """

        batch_size, current_seq_len = tgt.shape  # [64,1]

        pos_encoding = self.positional_encoding[:, :current_seq_len, :].to(tgt.device)  # torch.Size([1, 1, 128])

        tgt_embedded = self.embedding(tgt) + pos_encoding  # [batch_size, current_seq_len, d_model]

        if self.use_cf:
            tgt_embedded += self.class_embed(class_label)

        tgt_mask = self.generate_square_subsequent_mask(current_seq_len).to(tgt.device)

        memory = z.unsqueeze(0).expand(current_seq_len, batch_size, self.d_model)

        output = self.transformer_decoder(
            tgt_embedded.transpose(0, 1),
            memory,
            tgt_mask=tgt_mask
        ).transpose(0, 1)  # [batch_size, current_seq_len, d_model]

        return self.output_layer(output)  # [batch_size, current_seq_len, num_categories]

    def forward(self, src, class_label):
        """
        Args:
            src: [batch_size, seq_len]
            class_label: [batch_size, 1]
        """

        src = torch.cat([torch.full((src.shape[0], 1), self.num_categories,
                                    dtype=src.dtype, device=src.device), src], dim=1)       # b*(seq_len+1)
        mu, logvar = self.encode(src[:, 1:], class_label)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, src[:, :-1], class_label), mu, logvar

    def sample(self, num_samples, class_label=None):
        z = torch.randn(num_samples, self.d_model).to(next(self.parameters()).device)       # [64,128]
        start_token = torch.ones(num_samples, 1).long().to(z.device) * self.num_categories  # [64,1]
        generated = start_token

        for _ in range(self.seq_len):
            output = self.decode(z, generated, class_label)
            next_token = torch.distributions.Categorical(logits=output[:, -1, :]).sample()
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)

        return self.sequence_to_matrix(generated[:, 1:])

    def sequence_to_matrix(self, sequence):
        # sequence shape: [batch_size, seq_len]
        batch_size = sequence.shape[0]
        matrix = torch.zeros((batch_size, self.nf, self.nf), dtype=sequence.dtype).to(sequence.device)
        idx = torch.triu_indices(self.nf, self.nf, offset=1)
        matrix[:, idx[0], idx[1]] = sequence
        matrix = matrix + matrix.transpose(-2, -1) - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))
        return matrix


class GraphFeatModel(nn.Module):
    def __init__(self, input_face_features=8, num_face_features=16):
        super(GraphFeatModel, self).__init__()
        self.conv1 = GCNConv(input_face_features, num_face_features)
        self.conv2 = GCNConv(num_face_features, num_face_features)

    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: A tensor of shape [num_nodes, num_node_features].
            edge_index: A tensor of shape [2, num_edges].
            edge_weight: A tensor of shape [num_edges,]."""

        x = self.conv1(x, edge_index, edge_weight)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class TopoEncoder(nn.Module):
    def __init__(self, input_dim=16, d_model=256, num_head=4, dim_feedforward=256, dropout=0.1, n_layers=4):
        super().__init__()

        self.embed_layer = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, edge_embed, edge_mask, conds):
        """
        Args:
            edge_embed: A tensor of shape [batch_size, ne+2, m].
            edge_mask: A tensor of shape [batch_size, ne+2].
            conds: Condition embeddings, [batch_size, ?, d_model]
        Returns:
            A tensor of shape [batch_size, ne+2, d_model]."""

        # edge_embed = self.embed_layer(edge_embed)         # b*(ne+2)*d_model

        if conds is not None:
            edge_embed += conds                           # b*(ne+2)*d_model

        output = self.model(edge_embed.permute(1, 0, 2), src_key_padding_mask=~edge_mask)  # (ne+2)*b*d_model

        # Transpose back to b*(ne+2)*d_model
        output = output.permute(1, 0, 2)

        return output


class TopoDecoder(nn.Module):
    def __init__(self, max_seq_length, d_model=256, num_head=4, dim_feedforward=256, dropout=0.1, n_layers=4):
        super(TopoDecoder, self).__init__()

        self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.model = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        """
        Generates a target mask for the input sequence

        Args:
            sz: The Input Sequence Length.
        Returns:
            mask: A lower triangular matrix of shape [sequence_length, sequence_length].
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

        # """Generate a square mask for the sequence, where the mask prevents attention to future positions."""
        # mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        # mask = mask.masked_fill(mask, float('-inf'))
        # return mask

    def forward(self, seq_embed, seq_mask, sequential_context_embeddings, edge_mask):
        """
        Args:
            seq_embed: A tensor with shape [batch_size, seq_len, d_model].
            seq_mask: A tensor with shape [batch_size, seq_len].
            sequential_context_embeddings: A tensor with shape [batch_size, ne+2, d_model].
            edge_mask: A tensor with shape [batch_size, ne+2].
        Returns:
            A tensor with shape [batch_size, seq_len, d_model].
        """
        # assert seq_embed.shape[1] <= 390
        seq_embed += self.pos_embedding(torch.arange(seq_embed.shape[1],
                                                     device=seq_embed.device)).type_as(seq_embed)   # b*ns*d_model
        # Creating masks
        tgt_mask = self._generate_square_subsequent_mask(seq_embed.shape[1], device=seq_embed.device)

        # Forward pass through the Transformer decoder
        output = self.model(
            tgt=seq_embed.permute(1, 0, 2),                         # [seq_len, batch_size, d_model]
            memory=sequential_context_embeddings.permute(1, 0, 2),  # [ne+2, batch_size, d_model]
            tgt_mask=tgt_mask,                                      # [seq_len, seq_len]
            tgt_key_padding_mask=~seq_mask,                         # [batch_size, seq_len]
            memory_key_padding_mask=~edge_mask                      # [batch_size, ne+2]
        )

        # Transpose back to [batch_size, seq_len, d_model]
        output = output.permute(1, 0, 2)

        return output


class EdgeVertModel(nn.Module):
    def __init__(self, input_face_features=8, emb_features=16, d_model=256,
                 edge_classes=5, max_face=50, max_edge=30, max_num_edge=100, max_seq_length=1000, n_layers=4, use_cf=False):
        super(EdgeVertModel, self).__init__()

        # self.feat_extractor = GraphFeatModel(input_face_features=input_face_features,
        #                                      num_face_features=emb_features)
        # self.face_nEdge_embedding = nn.Embedding(max_edge, emb_features)
        emb_features = d_model
        self.face_idx_embedding = nn.Embedding(max_face, emb_features)
        self.edge_idx_embedding = nn.Embedding(max_num_edge, emb_features)
        self.edge_classes = edge_classes
        self.max_edge = max_edge
        self.max_num_edge = max_num_edge

        self.even_embed = nn.Parameter(torch.randn(emb_features))
        self.odd_embed = nn.Parameter(torch.randn(emb_features))
        self.loop_end_embed = nn.Parameter(torch.randn(emb_features))
        self.face_end_embed = nn.Parameter(torch.randn(emb_features))
        self.encoder = TopoEncoder(input_dim=emb_features, d_model=d_model, n_layers=n_layers)
        self.decoder = TopoDecoder(max_seq_length=max_seq_length, d_model=d_model, n_layers=n_layers)
        self.project_to_pointers = nn.Linear(d_model, d_model)

        self.fef_embedding = nn.Embedding(edge_classes, emb_features)

        self.use_cf = use_cf
        if self.use_cf:
            self.class_embedding = Embedder(11, d_model)

        self.cache = {}

    @staticmethod
    def get_edge_index(edgeFace_adj, edge_mask):
        batch_size, ne, _ = edgeFace_adj.shape

        face_offsets = torch.cumsum(torch.max(edgeFace_adj.reshape(batch_size, -1), dim=1)[0]+1, dim=0)
        face_offsets = torch.cat([torch.tensor([0], device=edgeFace_adj.device), face_offsets[:-1]])

        edgeFace_adj_offset = edgeFace_adj + face_offsets.reshape(-1, 1, 1)

        edge_index = edgeFace_adj_offset[edge_mask]

        return edge_index    # Ne*2

    def graph_feat_extra(self, edgeFace_adj, edge_mask, share_id):
        """
        Args:
            edgeFace_adj: A tensor of shape [batch_size, ne, 2].
            edge_mask: A tensor of shape [batch_size, ne].
            share_id: A tensor of shape [batch_size, ne].
        """

        batch_size, ne, _ = edgeFace_adj.shape
        device = edgeFace_adj.device

        edge_feat = self.face_idx_embedding(edgeFace_adj).mean(-2)       # b*ne*embed_features
        edge_feat += self.fef_embedding(share_id)                        # b*ne*embed_features

        edge_embed = torch.zeros((batch_size, edge_mask.shape[1] * 2, edge_feat.size(-1)),
                                 dtype=edge_feat.dtype, device=edge_feat.device)  # b*ne*m
        edge_embed[:, 0::2, :] = (edge_feat+self.even_embed)*edge_mask.unsqueeze(-1)
        edge_embed[:, 1::2, :] = (edge_feat+self.odd_embed)*edge_mask.unsqueeze(-1)
        edge_num_per = edge_mask.sum(1)                                  # b
        edge_num_per *= 2                                                # b

        edge_mask = make_mask(mask=edge_num_per.unsqueeze(-1), n=edge_embed.shape[1])

        # # compute face feature
        # edge_index = self.get_edge_index(edgeFace_adj, edge_mask)    # Ne*2
        # assert (edge_index[:, 0] - edge_index[:, 1]).abs().min() > 0
        # edge_index_unique, edge_index_num = torch.unique(
        #     torch.sort(edge_index, dim=1).values, dim=0, return_counts=True)
        #
        # assert edge_index_num.max() < self.edge_classes
        #
        # # assign edge feature
        # edge_num_per = edge_mask.sum(1)                                      # b
        # face_num_per = torch.max(edgeFace_adj.flatten(1, 2), dim=1)[0] + 1   # b
        # face_feat = self.face_idx_embedding(torch.cat(
        #     [torch.arange(i.item(), device=edgeFace_adj.device) for i in face_num_per]))
        # edge_feat = face_feat[edge_index].mean(1)    # Ne*embed_features
        #
        # edge_feat += self.fef_embedding(torch.cat(
        #     [torch.arange(i.item()) for i in edge_index_num]).to(edge_index_num.device))
        #
        # edge_feat = torch.cat(
        #     (edge_feat+self.even_embed.unsqueeze(0), edge_feat+self.odd_embed.unsqueeze(0)), dim=-1).reshape(
        #     2 * edge_feat.size(0), -1)               # (2*Ne)*embed_features
        #
        # # gather edges to batch
        # edge_num_per *= 2
        # edge_embed = torch.zeros((batch_size, edge_mask.shape[1]*2, edge_feat.size(1)),
        #                          dtype=edge_feat.dtype, device=edge_feat.device)                    # b*ne*m
        # edge_mask = torch.zeros((batch_size, edge_mask.shape[1]*2),
        #                         dtype=torch.bool, device=edge_feat.device)                          # b*ne
        #
        # start_indices = torch.cat([torch.zeros(1, device=edge_num_per.device, dtype=torch.long),
        #                            edge_num_per.cumsum(0)[:-1]], dim=0)
        # max_edges = edge_mask.shape[1]
        # range_tensor = torch.arange(max_edges, device=edge_feat.device)[None, :]
        # valid_mask = range_tensor < edge_num_per[:, None]
        # edge_mask[valid_mask] = True
        # indices_expand = range_tensor + start_indices[:, None]
        # edge_embed[valid_mask] = edge_feat[indices_expand[valid_mask]]

        temp = edge_embed.abs().sum(-1) > 0
        assert torch.equal(temp.sum(-1), edge_num_per)

        return edge_embed, edge_mask, edge_num_per

    def encoding(self, edge_embed, edge_mask, class_label):
        batch_size = edge_mask.shape[0]
        edge_embed = torch.cat([
            self.face_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, edge_embed.shape[-1]),
            self.loop_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, edge_embed.shape[-1]), edge_embed],
            dim=1)                                                                 # b*(ne+2)*m

        if self.use_cf:
            class_embeddings = self.class_embedding(class_label)                   # b*1*d_model
        else:
            class_embeddings = None

        edge_mask = torch.nn.functional.pad(edge_mask, [2, 0, 0, 0], value=1)                           # b*(ne+2)
        edge_embed = self.encoder(edge_embed=edge_embed, edge_mask=edge_mask, conds=class_embeddings)   # b*(ne+2)*d_model
        return edge_embed, edge_mask

    def forward(self, edgeFace_adj, edge_mask, topo_seq, seq_mask, share_id, class_label):
        """
        Args:
            edgeFace_adj: A tensor of shape [batch_size, ne, 2].
            edge_mask: A tensor of shape [batch_size, ne].
            topo_seq: A tensor of shape [batch_size, ns].
            seq_mask: A tensor of shape [batch_size, ns].
            share_id: A tensor of shape [batch_size, ne].
            class_label: A tensor of shape [batch_size, 1].
        Returns:
            A tensor with shape [batch_size, ns, ne+2]."""

        # ====================================Extract graph features================================================
        edge_embed, edge_mask, edge_num_per = self.graph_feat_extra(edgeFace_adj, edge_mask, share_id)
        assert torch.all(edge_mask.sum(-1) // 2 - 1 == topo_seq.max(-1)[0] // 2)

        # =====================================Transformer Encoding=================================================
        edge_embed, edge_mask = self.encoding(edge_embed, edge_mask, class_label)

        # =====================================Transformer Decoding=================================================
        batch_size = edge_mask.shape[0]
        sequential_context_embeddings = edge_embed * edge_mask.unsqueeze(-1)                  # b*(ne+2)*d_model
        seq_embed = torch.zeros((batch_size, topo_seq.shape[-1], edge_embed.shape[-1]),
                                device=edge_embed.device, dtype=edge_embed.dtype)             # b*ns*d_model
        for i in range(batch_size):
            seq_embed[i] = edge_embed[i, :edge_num_per[i]+2][topo_seq[i]+2]

        # if self.use_cf:
        #     seq_embed += self.class_embedding(class_label)

        outs = self.decoder(seq_embed, seq_mask, sequential_context_embeddings, edge_mask)   # b*ns*d_model

        # ========================================Pointer Logits===================================================
        pred_pointers = self.project_to_pointers(outs)                  # b*ns*d_model
        edge_embed_transposed = edge_embed.transpose(1, 2)              # b*d_model*(ne+2)
        logits = torch.matmul(pred_pointers, edge_embed_transposed)
        logits = logits / math.sqrt(pred_pointers.shape[-1])            # b*ns*(ne+2)
        logits = logits * edge_mask.unsqueeze(1)
        logits = logits - (~edge_mask.unsqueeze(1)).float() * 1e9       # b*ns*(ne+2)

        return logits

    def save_cache(self, edgeFace_adj, edge_mask, share_id, class_label):
        """
        Args:
            edgeFace_adj: A tensor of shape [batch_size, ne, 2].
            edge_mask: A tensor of shape [batch_size, ne].
            share_id: A tensor of shape [batch_size, ne].
            class_label: A tensor of shape [batch_size, 1].
        """

        # ====================================Extract graph features================================================
        edge_embed, edge_mask, edge_num_per = self.graph_feat_extra(edgeFace_adj, edge_mask, share_id)

        # =====================================Transformer Encoding=================================================
        edge_embed, edge_mask = self.encoding(edge_embed, edge_mask, class_label)
        sequential_context_embeddings = edge_embed * edge_mask.unsqueeze(-1)  # b*(ne+2)*d_model

        self.cache['edge_embed'] = edge_embed
        self.cache['edge_mask'] = edge_mask
        self.cache['sequential_context_embeddings'] = sequential_context_embeddings
        self.cache['edge_num_per'] = edge_num_per

    def clear_cache(self):
        self.cache.clear()

    def sample(self, topo_seq, seq_mask, mask, class_label):
        """
        Args:
            topo_seq: A tensor of shape [batch_size, ns].
            seq_mask: A tensor of shape [batch_size, ns].
            mask:
            class_label: [batch_size, 1]
        """

        # =====================================Transformer Decoding=================================================
        batch_size = topo_seq.shape[0]
        assert batch_size == 1
        edge_embed = self.cache['edge_embed']                                                 # b*(ne+2)*d_model
        edge_mask = self.cache['edge_mask']                                                   # b*(ne+2)
        sequential_context_embeddings = self.cache['sequential_context_embeddings']           # b*(ne+2)*d_model
        edge_num_per = self.cache['edge_num_per']
        seq_embed = torch.zeros((batch_size, topo_seq.shape[-1], edge_embed.shape[-1]),
                                device=edge_embed.device, dtype=edge_embed.dtype)             # b*ns*d_model
        for i in range(batch_size):
            seq_embed[i] = edge_embed[i, :edge_num_per[i]+2][topo_seq[i]+2]

        if self.use_cf:
            seq_embed += self.class_embedding(class_label)

        outs = self.decoder(seq_embed, seq_mask, sequential_context_embeddings, edge_mask)    # b*ns*d_model

        # ======================================Pointer Logits=====================================================
        pred_pointers = self.project_to_pointers(outs)[:, [-1], :]      # b*1*d_model
        edge_embed_transposed = edge_embed.transpose(1, 2)              # b*d_model*(ne+2)
        logits = torch.matmul(pred_pointers, edge_embed_transposed)
        logits = logits / math.sqrt(pred_pointers.shape[-1])            # b*1*(ne+2)
        logits = logits * edge_mask.unsqueeze(1)
        logits = logits - (~edge_mask.unsqueeze(1)).float() * 1e9       # b*1*(ne+2)

        logits = logits.squeeze()[mask]                                 # len(mask)

        return logits
