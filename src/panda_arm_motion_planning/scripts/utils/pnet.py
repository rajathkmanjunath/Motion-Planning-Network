from torch import nn


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fulcon = nn.Sequential(nn.Linear(input_size, 9600), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(9600, 7200), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(7200, 4800), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(4800, 3600), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(3600, 2400), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(2400, 1200), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(1200, 800), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(800, 768), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(384, 384), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(128, 128), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(64, 32), nn.PReLU(),
                                    nn.Linear(32, output_size))

    def forward(self, intensor):
        return self.fulcon(intensor)
