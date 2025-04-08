from diffusers.utils import logging

logger = logging.get_logger(__name__)


"""
`定义了`TransformerDiffusionMixin`类,为 TripoSG 系统中的扩散模型管道提供了重要的内存优化和性能增强功能。用于辅助带有VAE和变压器的扩散管道(主要针对DIT)。
类中提供了启用和禁用切片、平铺VAE解码以及融合和不融合QKV投影的方法。还包含用于日志记录的`logger`。 
"""
class TransformerDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and transformer.(mainly for DIT)
    """

    #VAE切片功能
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    #VAE平铺功能
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    #QKV 投影融合
    def fuse_qkv_projections(self, transformer: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            transformer (`bool`, defaults to `True`): To apply fusion on the Transformer.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_transformer = False
        self.fusing_vae = False

        if transformer:
            self.fusing_transformer = True
            self.transformer.fuse_qkv_projections()

        if vae:
            self.fusing_vae = True
            self.vae.fuse_qkv_projections()

    #取消QKV投影融合
    def unfuse_qkv_projections(self, transformer: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            transformer (`bool`, defaults to `True`): To apply fusion on the Transformer.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if transformer:
            if not self.fusing_transformer:
                logger.warning(
                    "The UNet was not initially fused for QKV projections. Doing nothing."
                )
            else:
                self.transformer.unfuse_qkv_projections()
                self.fusing_transformer = False

        if vae:
            if not self.fusing_vae:
                logger.warning(
                    "The VAE was not initially fused for QKV projections. Doing nothing."
                )
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False
