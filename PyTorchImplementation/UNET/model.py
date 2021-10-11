import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self, in_chans, out_chans, features=[64, 128, 256, 512]
        ):
        super(UNet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downward
        for f in features:
            self.downs.append( DoubleConv(in_chans,f) )            
            in_chans = f
        
        # upward
        for f in reversed(features):
            self.ups.append( 
                nn.ConvTranspose2d( f*2,f, kernel_size=2, stride=2 ) 
                )
            self.ups.append( DoubleConv(f*2,f) )   
                
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_chans, kernel_size=1)
        
    def forward(self, x):

        skip_conns = []

        for d in self.downs:
            x = d(x)
            skip_conns.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_conns = skip_conns[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_conn = skip_conns[i//2]

            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])

            concat_skip = torch.cat( (skip_conn, x), dim=1)
            x = self.ups[i+1](concat_skip)
        
        return self.final_conv(x)


# def test():
#     x = torch.randn((3,1,160,160))
#     model = UNet(in_chans=1, out_chans=1)
#     preds = model(x)
#     print(x.shape)
#     print(preds.shape)
#     assert preds.shape == x.shape

# if __name__ == "__main__":
#     test()

