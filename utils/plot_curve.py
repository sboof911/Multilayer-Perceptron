import matplotlib.pyplot as plt

class plot_curve:
    def __init__(self) -> None:
        self._loss_array = []
        self._acurracy_aray = []

    @property
    def loss_array(self):
        return self._loss_array

    @property
    def acurracy_aray(self):
        return self._acurracy_aray

    def plot_loss(self, filename='loss_curve.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, len(self._loss_array)), self._loss_array, label='Loss', color='red', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_accuracy(self, filename='accuracy_curve.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, len(self._acurracy_aray)), self._acurracy_aray, label='Accuracy', color='blue', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def save(self):
        self.plot_loss()
        self.plot_accuracy()
