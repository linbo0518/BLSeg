import torch
from torch import nn


class BackboneBaseModule(nn.Module):

    def __init__(self):
        super(BackboneBaseModule, self).__init__()
        self.strides = [2, 4, 8, 16, 32]

    def load_parameters(self, filename, map_location=None, strict=True):
        self.load_state_dict(torch.load(filename, map_location=map_location),
                             strict=strict)

    def change_output_stride(self, output_stride):
        assert output_stride in [8, 16, 32]
        if output_stride == 32:
            self._change_downsample([2, 2])
            self.strides[3] = 16
            self.strides[4] = 32
        elif output_stride == 16:
            self._change_downsample([2, 1])
            self.strides[3] = 16
            self.strides[4] = 16
        elif output_stride == 8:
            self._change_downsample([1, 1])
            self.strides[3] = 8
            self.strides[4] = 8

    def change_dilation(self, params):
        assert isinstance(params, (tuple, list))
        assert len(params) == 5
        self._change_stage_dilation(self.stage0, params[0])
        self._change_stage_dilation(self.stage1, params[1])
        self._change_stage_dilation(self.stage2, params[2])
        self._change_stage_dilation(self.stage3, params[3])
        self._change_stage_dilation(self.stage4, params[4])

    def _change_stage_dilation(self, stage, param):
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (param, param)
                    m.dilation = (param, param)

    def _change_downsample(self, params):
        '''
        Should be overridden by all subclasses
        '''
        raise NotImplementedError

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)