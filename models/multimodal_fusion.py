import torch

from torch import nn
from torch.autograd import Variable


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.downsamplex_2 = nn.AvgPool2d(2, 2)
        self.downsamplex_3 = nn.AvgPool2d(2, 2)

        self.downsampley_2 = nn.AvgPool2d(2, 2)
        self.downsampley_3 = nn.AvgPool2d(2, 2)

        self.deconv1_2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4,
                                     stride=2, padding=1, output_padding=0, bias=False)
        self.deconv2_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                                          stride=1, padding=0, output_padding=0, bias=False)
        self.deconv1_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                               stride=1, padding=0, output_padding=0, bias=False),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4,
                           stride=2, padding=1, output_padding=0, bias=False)
        )

        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv1 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.average_pooling2_1 = nn.AvgPool2d(2, 2)
        self.average_pooling3_2 = nn.AvgPool2d(2, 2)

        self.average_pooling3_1 = nn.Sequential(nn.AvgPool2d(2, 2), nn.AvgPool2d(2, 2))

    def forward(self, x, y):
        x1, y1 = x, y
        x2, y2 = self.downsamplex_2(x1), self.downsampley_2(y1)
        x3, y3 = self.downsamplex_3(x2), self.downsampley_3(y2)

        hid_11 = self.conv1_1(torch.cat((x1, y1), 1))
        # hid_12 = self.deconv1_2(y2)
        hid_12 = self.conv1_2(torch.cat((x1, self.deconv1_2(y2)), 1))
        hid_13 = self.conv1_3(torch.cat((x1, self.deconv1_3(y3)), 1))

        hid_21 = self.conv2_1(torch.cat((x2, self.average_pooling2_1(y1)), 1))
        hid_22 = self.conv2_2(torch.cat((x2, y2), 1))
        hid_23 = self.conv2_3(torch.cat((x2, self.deconv2_3(y3)), 1))

        hid_31 = self.conv3_1(torch.cat((x3, self.average_pooling3_1(y1)), 1))
        hid_32 = self.conv3_2(torch.cat((x3, self.average_pooling3_2(y2)), 1))
        hid_33 = self.conv3_3(torch.cat((x3, y3), 1))

        res_1 = torch.cat((hid_11, hid_12, hid_13), 1)
        res_2 = torch.cat((hid_21, hid_22, hid_23), 1)
        res_3 = torch.cat((hid_31, hid_32, hid_33), 1)

        res_1 = self.conv1(res_1)
        res_2 = self.conv2(res_2)
        res_3 = self.conv3(res_3)

        return res_1, res_2, res_3


def main():
    large = (1, 128, 8, 8)
    middle = (1, 128, 4, 4)
    small = (1, 128, 2, 2)
    x = Variable(torch.randn(large))
    y = Variable(torch.randn(large))
    print(x.shape)
    fusion = Fusion()
    result1, result2, result3 = fusion(x, y)
    print("output size：")
    print(result1.shape)
    print(result2.shape)
    print(result3.shape)


if __name__ == "__main__":
    main()
