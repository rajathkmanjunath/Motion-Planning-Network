from torch import nn


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fulcon = nn.Sequential(nn.Linear(input_size, 1024), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(1000, 9000), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(9000, 7200), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(7200, 6400), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(6400, 5600), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(5600, 4800), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(4800, 4200), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(4200, 3600), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(3600, 3000), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(3000, 2400), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(2400, 1280), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(896, 800), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(800, 768), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(768, 720), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(720, 512), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(512, 480), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(480, 480), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(480, 384), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(384, 384), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                                    # nn.Linear(128, 128), nn.PReLU(),
                                    nn.Linear(128, 64), nn.PReLU(),
                                    nn.Linear(64, 64), nn.PReLU(),
                                    nn.Linear(64, 32), nn.PReLU(),
                                    nn.Linear(32, output_size))

    def forward(self, intensor):
        return self.fulcon(intensor)
