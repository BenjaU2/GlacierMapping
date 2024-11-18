import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any

######################################### DeepLabV3PLus ###########################################
class SEModule(nn.Module):
    """ El módulo Squeeze-and-Excitation (SE) tiene como objetivo recalibrar los canales de las características aprendidas, 
    es decir, ajustar la importancia de cada canal en un tensor de características.
    Aunque el tensor y final tiene la misma forma (32, 64, 1, 1) que el tensor después del average pooling, 
    el valor de cada canal ahora ha sido modificado para reflejar la importancia de ese canal en relación con los demás. 
    Este ajuste no ocurriría si simplemente usaras el tensor inmediatamente después del average pooling sin pasar por las capas Linear.
    En resumen, el propósito de este proceso es aprender a qué canales prestar más atención y cuáles suprimir, 
    lo que mejora la capacidad de la red para enfocar la información relevante en cada canal.

 """
    
    def __init__(self, channels: int, ratio: int = 8) -> None:
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print(x.shape)
        y = self.avgpool(x).view(b, c)
        #print(self.avgpool(x).shape)
        #print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        #print(y.shape)
        return x * y
      

# Clase contenedora para aplicar el SE Module y visualizar el proceso
class TestSEModule(nn.Module):
    def __init__(self):
        super(TestSEModule, self).__init__()
        self.se = SEModule(channels=64, ratio=8)
    
    def forward(self, x):
        return self.se.forward(x)
"""

# Verificar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear un tensor de prueba y moverlo al dispositivo
input_tensor = torch.randn(32, 64, 10, 10).to(device)

# Instanciar el modelo y moverlo al dispositivo
model1 = TestSEModule().to(device)
# Usar torchsummary
summary(model1, input_tensor.shape[1:])
#print(model)
"""
    
class ASPPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int]) -> None:
        super(ASPPModule, self).__init__()

        #Atrous convolutions
        self.atrous_convs = nn.ModuleList()
        for d in dilations:
            at_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, dilation=d, padding="same", bias=False
            )
            self.atrous_convs.append(at_conv)
        
        
        #self.avgpool = nn.AvgPool2d(kernel_size=(16,16))
        #self.avgpool = nn.AvgPool2d(kernel_size=(1,1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Asegura que el tamaño sea compatible
        self.batch_norm = nn.BatchNorm2d(out_channels)
        #self.batch_norm = nn.InstanceNorm2d(out_channels)  # Reemplazar BatchNorm por InstanceNorm
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.squeeze_excite = SEModule(channels = out_channels)
        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same", bias=False)
        # Final_convolution
        self.final_conv = nn.Conv2d(in_channels = out_channels*(len(dilations)+2), out_channels=out_channels, kernel_size=1, padding="same", bias=False)

    def forward(self, x: Any) -> Any:
        # ASPP forward

        # 1x1 conv
        x1 = self.conv1x1(x)
        x1 = self.batch_norm(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x1 = self.squeeze_excite(x1)

        # Atrous Convolutions
        atrous_outputs = []
        for at_conv in self.atrous_convs:
            at_output = at_conv(x)
            at_output = self.batch_norm(at_output)
            at_output = self.relu(at_output)
            at_output = self.squeeze_excite(at_output)
            atrous_outputs.append(at_output)
        
        # Global Average Pooling and 1x1 COnvolution
        avg_pool = self.avgpool(x)
        avg_pool = self.conv1x1(avg_pool)
        #avg_pool = self.batch_norm(avg_pool)
        avg_pool = self.relu(avg_pool)
        #avg_pool = self.upsample(avg_pool)
        avg_pool = torch.nn.functional.interpolate(avg_pool, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        avg_pool = self.squeeze_excite(avg_pool)

        # Concatenating Dilated convolutions and Gloval Averge Pooling
        combined_output = torch.cat((x1, *atrous_outputs, avg_pool), dim = 1)

        # Final convolution
        assp_output = self.final_conv(combined_output)
        assp_output = self.batch_norm(assp_output)
        assp_output = self.relu(assp_output)
        assp_output = self.squeeze_excite(assp_output)

        return assp_output

class DecoderModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderModule, self).__init__()
        # squeeze and excite module
        self.squeeze_excite = SEModule(channels=304)
        self.squeeze_excite2 = SEModule(channels=out_channels)
        self.squeeze_excite3 = SEModule(channels=48)

        #  1x1 conv
        self.conv_low = nn.Conv2d(in_channels, 48, kernel_size=1, padding="same", bias = False)

        self.batch_norm = nn.BatchNorm2d(48)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # 3x3 conv
        self.final_conv1 = nn.Conv2d(
            in_channels=304, out_channels = 256, kernel_size=3, padding="same", bias=False
        )

        # 3x3conv
        self.final_conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def forward(self, x_high: Any, x_low: Any) -> Any:
        # 1x1 Convolution on Low level feature
        x_low = self.conv_low(x_low)
        x_low = self.batch_norm(x_low)
        x_low = self.dropout(x_low)
        x_low = self.relu(x_low)
        x_low = self.squeeze_excite3(x_low)

        # Concatenating High-level and Low Level feature
        x = torch.cat((x_high, x_low), dim=1)
        x = self.dropout(x)
        x = self.squeeze_excite(x)

        # 3x3 convolution on concatening feature map
        x = self.final_conv1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.squeeze_excite2(x)

        # 3x3 convolution 
        x = self.final_conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.squeeze_excite2(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 6) -> None:
        super(DeepLabV3Plus, self).__init__()
        # Load pretained ResNet model
        resnet =models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        # Modify the first conv layer to accept in_channels to 6 channels to input
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize weights of the modified first layer
        nn.init.kaiming_normal_(self.backbone[0].weight, mode="fan_out", nonlinearity="relu")

       
        out_channels = 256

        # Dilations rate 
        dilations = [6, 12, 18, 24]

        # ASPPModule
        self.aspp = ASPPModule(1024, out_channels=out_channels, dilations=dilations)

        # Decoder model
        self.decoder = DecoderModule(out_channels, out_channels)

        # Upsampling with bilinear interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        
        # Dropout
        self.dropout =nn.Dropout(p=0.5)

        # final convolutional
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        #Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Any) -> Any:
        # Getting Low-Level Features
        x_low = self.backbone[:-3](x)
        # Getting Image Features from Backbone
        x = self.backbone[:-1](x)
        # ASPP forward pass - High-Level Features
        x = self.aspp(x)
        # Upsampling High-Level Features
        x = self.upsample(x)
        x = self.dropout(x)
         # Decoder forward pass - Concatenating Features
        x = self.decoder(x, x_low)

        # Upsampling Concatenated Features from Decoder
        x = self.upsample(x)
        # Final 1x1 Convolution for Binary-Segmentation
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x