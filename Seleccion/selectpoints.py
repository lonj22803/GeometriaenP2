import matplotlib.pyplot as plt

class SeleccionDePuntos:
    def __init__(self, num_puntos_minimos=5):
        self.num_puntos_minimos = num_puntos_minimos
        self.points = []

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        plt.grid(True)
        plt.title(f'Seleccion de Puntos\nUna vez seleccionados más de 5 puntos, presione la tecla r')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            self.points.append((x, y))
            plt.scatter(x, y, c='red')
            plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            plt.draw()

    def onkey(self, event):
        if event.key == 'r':
            plt.close(self.fig)
            if len(self.points) < self.num_puntos_minimos:
                print(f"Debe seleccionar al menos {self.num_puntos_minimos} puntos.")
            else:
                print("Puntos seleccionados:")
                for point in self.points:
                    print(point)

    def show(self):
        plt.show()

if __name__ == "__main__":
    seleccionador = SeleccionDePuntos(num_puntos_minimos=5)
    seleccionador.show()
    print(seleccionador)
    print(seleccionador.points)
    print(type(seleccionador.points))
    print(len(seleccionador.points))
