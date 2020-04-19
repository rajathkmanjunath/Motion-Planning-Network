from torch import nn


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fulcon = nn.Sequential(nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(64, 32), nn.PReLU(),
                                    nn.Linear(32, output_size))

    def forward(self, intensor):
        return self.fulcon(intensor)
