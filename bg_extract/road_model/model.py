import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models
from ..base import BaseModel
# from inplace_abn import InPlaceABN


# ========================================
# FuseSatDeepLabV3Plus
# - Satellite branch
# ========================================
# Module - FuseSatDeepLabV3PlusDecoder
class FuseSatDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            DeepLabASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        fms = [features[-1]]

        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)

        fms.append(features[-4])
        high_res_features = self.block1(features[-4])

        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fms.append(concat_features)
        fused_features = self.block2(concat_features)
        fms.append(fused_features)
        return fused_features, fms


# Module - ASPPConv
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


# Module - ASPPSeparableConv
class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


# Module - ASPPPooling
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# Module - DeepLabASPP
class DeepLabASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(DeepLabASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# Module - SeparableConv2d
class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


# Model - FuseSatDeepLabV3Plus
class FuseSatDeepLabV3Plus(BaseModel):
    def __init__(self, in_channels=3, encoder_name='resnet34', encoder_weights='imagenet', classes=1,
                 activation='sigmoid'):
        super(FuseSatDeepLabV3Plus, self).__init__()
        deeplab = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        self.encoder = deeplab.encoder
        self.decoder = FuseSatDeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )
        self.segmentation_head = deeplab.segmentation_head

    def forward(self, x):
        encs = self.encoder(x)
        decs, fms = self.decoder(*encs)
        probs = self.segmentation_head(decs)
        return probs, encs, fms


# ========================================
# FuseParDeepLabV3Plus
# - Partial branch
# ========================================
# Module - FuseParDeepLabV3PlusDecoder
class FuseParDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            DeepLabASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.adaptor1 = nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1], kernel_size=3, stride=1, padding=1)
        self.adaptor2 = nn.Conv2d(highres_in_channels * 2, highres_in_channels, kernel_size=3, stride=1, padding=1)
        self.adaptor3 = nn.Conv2d((highres_out_channels + out_channels) * 2, highres_out_channels + out_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, fms, *features):
        x = features[-1]
        x = torch.cat([x, fms[0]], dim=1)
        x = self.adaptor1(x)
        fms_out = [x]
        aspp_features = self.aspp(x)
        aspp_features = self.up(aspp_features)

        x = features[-4]
        x = torch.cat([x, fms[1]], dim=1)
        x = self.adaptor2(x)
        fms_out.append(x)
        high_res_features = self.block1(x)

        x = torch.cat([aspp_features, high_res_features], dim=1)
        x = torch.cat([x, fms[2]], dim=1)
        x = self.adaptor3(x)
        fms_out.append(x)
        fused_features = self.block2(x)
        fms_out.append(fused_features)
        return fused_features, fms_out


# Model - FuseParDeepLabV3Plus
class FuseParDeepLabV3Plus(BaseModel):
    def __init__(self, in_channels=3, encoder_name='resnet34', encoder_weights='imagenet', classes=1,
                 activation='sigmoid'):
        super(FuseParDeepLabV3Plus, self).__init__()
        deeplab = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        self.encoder = deeplab.encoder
        self.adaptor1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.adaptor2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.adaptor3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.adaptor4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.adaptors = [self.adaptor1, self.adaptor2, self.adaptor3, self.adaptor4]
        self.decoder = FuseParDeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )
        self.segmentation_head = deeplab.segmentation_head

    def forward(self, x, encs_in, fms_in):
        stages = self.encoder.get_stages()

        encs = []
        for i in range(self.encoder._depth + 1):
            # layer1    =>  2
            # layer2    =>  3
            # layer3    =>  4
            # layer4    =>  5
            if i >= 2:
                x = self.adaptors[i - 2](torch.cat([x, encs_in[i - 1]], dim=1))
            x = stages[i](x)
            encs.append(x)

        # encs = self.encoder(x)
        decs, fms_out = self.decoder(fms_in, *encs)
        probs = self.segmentation_head(decs)
        return probs, fms_out


# ========================================
# LateFuseSatDeepLabV3Plus
# - Satellite branch
# ========================================
# Model - LateFuseSatDeepLabV3Plus
class LateFuseSatDeepLabV3Plus(BaseModel):
    def __init__(self, in_channels=3, encoder_name='resnet34', encoder_weights='imagenet', classes=1,
                 activation=None):
        super(LateFuseSatDeepLabV3Plus, self).__init__()
        deeplab = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        self.encoder = deeplab.encoder
        self.decoder = FuseSatDeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )
        self.segmentation_head = deeplab.segmentation_head

        self.adaptor1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.adaptor2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, partial):
        encs = self.encoder(x)
        decs, fms = self.decoder(*encs)
        sat_probs = self.segmentation_head(decs)

        x = self.adaptor1(sat_probs)
        y = self.adaptor2(partial)
        probs = torch.cat([x, y], dim=1)
        probs = self.conv_fuse(probs)

        sat_probs = self.sigmoid(sat_probs)
        probs = self.sigmoid(probs)
        return sat_probs, probs


# ========================================
# DSAM
# - Dual Self-Attention Module
# ========================================
# Module - DSAM
class DSAM(nn.Module):
    def __init__(self, channels):
        super(DSAM, self).__init__()
        # adaptors
        self.s_adaptor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.c_adaptor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.out_adaptor = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # spatial attention
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1, stride=1, padding=0)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.s_gamma = nn.Parameter(torch.zeros(1))

        # channel attention
        self.c_gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # spatial attention
        s_x = self.s_adaptor(x)
        bs, c, h, w = s_x.size()
        proj_query = self.query_conv(x).view(bs, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(bs, -1, h * w)
        proj_value = self.value_conv(x).view(bs, -1, h * w)

        aff_map = torch.bmm(proj_query, proj_key)
        aff_map = self.softmax(aff_map)
        aff_map = torch.bmm(proj_value, aff_map.permute(0, 2, 1))

        aff_map = aff_map.view(bs, c, h, w)
        sa_map = self.s_gamma * aff_map + s_x

        # channel attention
        c_x = self.c_adaptor(x)
        bs, c, h, w = c_x.size()
        proj_query = c_x.view(bs, -1, h * w)
        proj_key = c_x.view(bs, -1, h * w).permute(0, 2, 1)
        proj_value = c_x.view(bs, -1, h * w)

        aff_map = torch.bmm(proj_query, proj_key)
        aff_map = torch.max(aff_map, -1, keepdim=True)[0].expand_as(aff_map) - aff_map
        aff_map = self.softmax(aff_map)
        aff_map = torch.bmm(aff_map, proj_value)

        aff_map = aff_map.view(bs, c, h, w)
        ca_map = self.c_gamma * aff_map + c_x

        # sum fusion
        a_map = sa_map + ca_map
        a_map = self.out_adaptor(a_map)
        return a_map


# ========================================
# GSAM
# - Gated Self-Attention Module
# ========================================
# Module - GSAM
class GSAM(nn.Module):
    def __init__(self, channels, use_att=True, use_gate=True, use_gap=False, use_pp=False, use_partial_loss=False):
        super(GSAM, self).__init__()
        # pyramid pooling
        self.use_pp = use_pp
        if use_pp:
            self.sat_pp = PP(channels)
            self.par_pp = PP(channels)

        # channel attention
        self.use_att = use_att
        if use_att:
            self.sat_gamma = nn.Parameter(torch.zeros(1))
            self.par_gamma = nn.Parameter(torch.zeros(1))

            self.sat_ca_map = None
            self.par_ca_map = None

        # gated fusion
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.use_gate = use_gate
        self.use_gap = use_gap
        if use_gate:
            # self.conv_block_2 = nn.Sequential(
            #     nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(channels // 2),
            #     nn.ReLU(inplace=True)
            # )
            # self.conv_block_3 = nn.Sequential(
            #     nn.Conv2d(channels // 2, 2, kernel_size=1, stride=1, padding=0),
            #     nn.ReLU(inplace=True)
            # )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(inplace=True)
            )
            self.conv_block_3 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(inplace=True)
            )

            self.sat_g = None
            self.par_g = None
            self.fuse_fm = None

        self.softmax = nn.Softmax(dim=-1)
        self.softmax_c = nn.Softmax(dim=1)

        self.use_partial_loss = use_partial_loss

    def forward(self, sat_fm, par_fm):
        # pyramid pooling
        if self.use_pp:
            sat_fm = self.sat_pp(sat_fm)
            par_fm = self.par_pp(par_fm)

        sat_ca_map = sat_fm
        par_ca_map = par_fm

        # channel attention
        if self.use_att:
            # satellite features
            bs, c, h, w = sat_fm.size()
            sat_proj_query = sat_fm.view(bs, -1, h * w)
            sat_proj_key = sat_fm.view(bs, -1, h * w).permute(0, 2, 1)
            sat_proj_value = sat_fm.view(bs, -1, h * w)

            sat_aff_map = torch.bmm(sat_proj_query, sat_proj_key)
            sat_aff_map = torch.max(sat_aff_map, -1, keepdim=True)[0].expand_as(sat_aff_map) - sat_aff_map
            sat_aff_map = self.softmax(sat_aff_map)

            # partial features
            bs, c, h, w = par_fm.size()
            par_proj_query = par_fm.view(bs, -1, h * w)
            par_proj_key = par_fm.view(bs, -1, h * w).permute(0, 2, 1)
            par_proj_value = par_fm.view(bs, -1, h * w)

            par_aff_map = torch.bmm(par_proj_query, par_proj_key)
            par_aff_map = torch.max(par_aff_map, -1, keepdim=True)[0].expand_as(par_aff_map) - par_aff_map
            par_aff_map = self.softmax(par_aff_map)

            # fuse affinity maps
            aff_map = sat_aff_map * par_aff_map

            # satellite features
            sat_aff_map = torch.bmm(aff_map, sat_proj_value)

            sat_aff_map = sat_aff_map.view(bs, c, h, w)
            sat_ca_map = self.sat_gamma * sat_aff_map + sat_fm

            # partial features
            par_aff_map = torch.bmm(aff_map, par_proj_value)
            par_aff_map = par_aff_map.view(bs, c, h, w)
            par_ca_map = self.par_gamma * par_aff_map + par_fm

            # # satellite features
            # sat_aff_map = torch.bmm(sat_aff_map, sat_proj_value)
            #
            # sat_aff_map = sat_aff_map.view(bs, c, h, w)
            # sat_ca_map = self.sat_gamma * sat_aff_map + sat_fm
            #
            # # partial features
            # par_aff_map = torch.bmm(par_aff_map, par_proj_value)
            # par_aff_map = par_aff_map.view(bs, c, h, w)
            # par_ca_map = self.par_gamma * par_aff_map + par_fm

            # # for visualization
            # self.sat_ca_map = sat_ca_map.detach()
            # self.par_ca_map = par_ca_map.detach()

        # gated fusion
        x = torch.cat([sat_ca_map, par_ca_map], dim=1)
        x = self.conv_block_1(x)
        if self.use_gate:
            # x = self.conv_block_2(x)
            # x = self.conv_block_3(x)
            # # x = self.softmax(x)
            # x = self.softmax_c(x)
            #
            # sat_g = x[:, 0, :, :]
            # sat_g = sat_g.unsqueeze(1)
            # par_g = x[:, 1, :, :]
            # par_g = par_g.unsqueeze(1)

            x1 = self.conv_block_2(x)
            x2 = self.conv_block_3(x)
            if self.use_gap:
                x1 = F.adaptive_avg_pool2d(x1, (1, 1))
                x2 = F.adaptive_avg_pool2d(x2, (1, 1))
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x = torch.cat([x1, x2], dim=1)
            x = self.softmax_c(x)

            sat_g = x[:, 0, :, :, :]
            par_g = x[:, 1, :, :, :]

            sat_ca_map = sat_ca_map * sat_g
            par_ca_map = par_ca_map * par_g
            fuse_fm = sat_ca_map + par_ca_map

            # # for visualization
            # self.sat_g = sat_g.detach()
            # self.par_g = par_g.detach()
            # self.fuse_fm = fuse_fm.detach()

            if self.use_partial_loss:
                return fuse_fm, par_ca_map
        else:
            fuse_fm = x
        return fuse_fm


# ========================================
# PP
# - PyramidPooling
# ========================================
# Module - PP
class PP(nn.Module):
    def __init__(self, channels, scales=(1, 2, 3, 6)):
        super(PP, self).__init__()
        self.blocks = nn.ModuleList([
            PPBlock(channels, channels // len(scales), scale) for scale in scales
        ])
        self.conv_out = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        x = self.conv_out(x)
        return x


# Module - PPBlock
class PPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=1):
        super(PPBlock, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


# ========================================
# FusePlusSatDeepLabV3Plus
# - Satellite branch
# ========================================
# Module - FusePlusSatDeepLabV3PlusDecoder
class FusePlusSatDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16, use_dsam=True,
                 share_dec=False, use_gsam=False, use_att=False, use_gate=False):
        super().__init__()
        self.share_dec = share_dec
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            DeepLabASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.use_dsam = use_dsam
        if use_dsam:
            self.dsam = DSAM(out_channels)

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if share_dec:
            self.use_gsam = use_gsam
            if use_gsam:
                self.adaptor1 = GSAM(encoder_channels[-1], use_att=use_att, use_gate=use_gate)
                self.adaptor2 = GSAM(highres_in_channels, use_att=use_att, use_gate=use_gate)
                self.adaptor3 = GSAM(highres_out_channels + out_channels, use_att=use_att, use_gate=use_gate)
            else:
                self.adaptor1 = nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1], kernel_size=3, stride=1,
                                          padding=1)
                self.adaptor2 = nn.Conv2d(highres_in_channels * 2, highres_in_channels, kernel_size=3, stride=1,
                                          padding=1)
                self.adaptor3 = nn.Conv2d((highres_out_channels + out_channels) * 2,
                                          highres_out_channels + out_channels,
                                          kernel_size=3, stride=1, padding=1)

    def forward(self, *features, is_par=False, fms=None):
        if not is_par:
            fms = [features[-1]]

            aspp_features = self.aspp(features[-1])
            if self.use_dsam:
                aspp_features = self.dsam(aspp_features)
            aspp_features = self.up(aspp_features)

            fms.append(features[-4])
            high_res_features = self.block1(features[-4])

            concat_features = torch.cat([aspp_features, high_res_features], dim=1)
            fms.append(concat_features)
            fused_features = self.block2(concat_features)
            fms.append(fused_features)
        else:
            x = features[-1]
            if self.use_gsam:
                x = self.adaptor1(fms[0], x)
            else:
                x = torch.cat([x, fms[0]], dim=1)
                x = self.adaptor1(x)
            fms_out = [x]
            aspp_features = self.aspp(x)
            if self.use_dsam:
                aspp_features = self.dsam(aspp_features)
            aspp_features = self.up(aspp_features)

            x = features[-4]
            if self.use_gsam:
                x = self.adaptor2(fms[1], x)
            else:
                x = torch.cat([x, fms[1]], dim=1)
                x = self.adaptor2(x)
            fms_out.append(x)
            high_res_features = self.block1(x)

            x = torch.cat([aspp_features, high_res_features], dim=1)
            if self.use_gsam:
                x = self.adaptor3(fms[2], x)
            else:
                x = torch.cat([x, fms[2]], dim=1)
                x = self.adaptor3(x)
            fms_out.append(x)
            fused_features = self.block2(x)
            fms_out.append(fused_features)
        return fused_features, fms


# Model - FusePlusSatDeepLabV3Plus
class FusePlusSatDeepLabV3Plus(BaseModel):
    def __init__(self, in_channels=3, encoder_name='resnet34', encoder_weights='imagenet', classes=1,
                 activation='sigmoid', use_dsam=False, share_dec=False, use_gsam=False, use_att=False, use_gate=False):
        super(FusePlusSatDeepLabV3Plus, self).__init__()
        self.share_dec = share_dec
        deeplab = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        self.encoder = deeplab.encoder
        self.decoder = FusePlusSatDeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            use_dsam=use_dsam,
            share_dec=share_dec,
            use_gsam=use_gsam,
            use_att=use_att,
            use_gate=use_gate
        )
        self.segmentation_head = deeplab.segmentation_head

    def forward(self, x):
        encs = self.encoder(x)
        decs, fms = self.decoder(*encs)
        probs = self.segmentation_head(decs)
        return probs, encs, fms


# ========================================
# FusePlusParDeepLabV3Plus
# - Partial branch
# ========================================
# Module - FusePlusParDeepLabV3PlusDecoder
class FusePlusParDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16, use_dsam=True,
                 use_gsam=True, use_att=True, use_gate=True, use_gap=False, use_pp=False, use_partial_loss=False):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            DeepLabASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.use_dsam = use_dsam
        if use_dsam:
            self.dsam = DSAM(out_channels)

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.use_gsam = use_gsam
        self.use_partial_loss = use_partial_loss
        if use_gsam:
            self.adaptor1 = GSAM(encoder_channels[-1], use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
            self.adaptor2 = GSAM(highres_in_channels, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
            self.adaptor3 = GSAM(highres_out_channels + out_channels, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp, use_partial_loss=use_partial_loss)
        else:
            self.adaptor1 = nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1], kernel_size=3, stride=1,
                                      padding=1)
            self.adaptor2 = nn.Conv2d(highres_in_channels * 2, highres_in_channels, kernel_size=3, stride=1, padding=1)
            self.adaptor3 = nn.Conv2d((highres_out_channels + out_channels) * 2, highres_out_channels + out_channels,
                                      kernel_size=3, stride=1, padding=1)

    def forward(self, fms, *features):
        x = features[-1]
        if self.use_gsam:
            x = self.adaptor1(fms[0], x)
        else:
            x = torch.cat([x, fms[0]], dim=1)
            x = self.adaptor1(x)
        fms_out = [x]
        aspp_features = self.aspp(x)
        if self.use_dsam:
            aspp_features = self.dsam(aspp_features)
        aspp_features = self.up(aspp_features)

        x = features[-4]
        if self.use_gsam:
            x = self.adaptor2(fms[1], x)
        else:
            x = torch.cat([x, fms[1]], dim=1)
            x = self.adaptor2(x)
        fms_out.append(x)
        high_res_features = self.block1(x)

        x = torch.cat([aspp_features, high_res_features], dim=1)
        if self.use_gsam:
            if self.use_partial_loss:
                x, par_ca_map = self.adaptor3(fms[2], x)
            else:
                x = self.adaptor3(fms[2], x)
        else:
            x = torch.cat([x, fms[2]], dim=1)
            x = self.adaptor3(x)
        fms_out.append(x)
        fused_features = self.block2(x)
        fms_out.append(fused_features)
        if self.use_partial_loss:
            par_ca_features = self.block2(par_ca_map)
            return fused_features, fms_out, par_ca_features
        return fused_features, fms_out


# Model - FusePlusParDeepLabV3Plus
class FusePlusParDeepLabV3Plus(BaseModel):
    def __init__(self, in_channels=3, encoder_name='resnet34', encoder_weights='imagenet', classes=1,
                 activation='sigmoid', use_dsam=False, use_gsam=True, use_att=True, use_gate=True, share_dec=False,
                 share_head=False, use_gap=False, use_pp=False, use_partial_loss=False):
        super(FusePlusParDeepLabV3Plus, self).__init__()
        self.share_dec = share_dec
        self.share_head = share_head
        self.use_partial_loss = use_partial_loss
        deeplab = smp.DeepLabV3Plus(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        self.encoder = deeplab.encoder
        if share_dec:
            self.decoder = nn.Identity()
        else:
            self.decoder = FusePlusParDeepLabV3PlusDecoder(
                encoder_channels=self.encoder.out_channels,
                out_channels=256,
                atrous_rates=(12, 24, 36),
                output_stride=16,
                use_dsam=use_dsam,
                use_gsam=use_gsam,
                use_att=use_att,
                use_gate=use_gate,
                use_gap=use_gap,
                use_pp=use_pp,
                use_partial_loss=use_partial_loss
            )
            if not share_head:
                self.segmentation_head = deeplab.segmentation_head

        self.use_gsam = use_gsam
        if use_gsam:
            self.adaptor1 = GSAM(64, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
            self.adaptor2 = GSAM(64, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
            self.adaptor3 = GSAM(128, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
            self.adaptor4 = GSAM(256, use_att=use_att, use_gate=use_gate, use_gap=use_gap, use_pp=use_pp)
        else:
            self.adaptor1 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.adaptor2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.adaptor3 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            self.adaptor4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.adaptors = [self.adaptor1, self.adaptor2, self.adaptor3, self.adaptor4]

    def forward(self, x, encs_in, fms_in):
        stages = self.encoder.get_stages()

        # # detach - seperate training
        # for idx in range(len(encs_in)):
        #     encs_in[idx] = encs_in[idx].detach()
        #
        # for idx in range(len(fms_in)):
        #     fms_in[idx] = fms_in[idx].detach()

        encs = []
        for i in range(self.encoder._depth + 1):
            # layer1    =>  2
            # layer2    =>  3
            # layer3    =>  4
            # layer4    =>  5
            if i >= 2:
                if self.use_gsam:
                    x = self.adaptors[i - 2](encs_in[i - 1], x)
                else:
                    x = torch.cat([x, encs_in[i - 1]], dim=1)
                    x = self.adaptors[i - 2](x)
            x = stages[i](x)
            encs.append(x)

        # encs = self.encoder(x)
        if not self.share_dec:
            if self.use_partial_loss:
                decs, fms_out, pars = self.decoder(fms_in, *encs)
                probs = self.segmentation_head(decs)
                probs_par = self.segmentation_head(pars)
                return probs, probs_par, fms_out
            else:
                decs, fms_out = self.decoder(fms_in, *encs)
                if not self.share_head:
                    probs = self.segmentation_head(decs)
                    return probs, fms_out
                else:
                    return decs
        else:
            return encs
