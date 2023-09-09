import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class HomogeneousGeometry(ABC):
    def __init__(self):
        self.elements = []

    def add_element(self, element, color):
        self.elements.append((element, color))

    def plot(self):
        # Crear una figura y configurar el aspecto igual (aspect='equal')
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')

        for element, color in self.elements:
            element.plot(color)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Grafico')
        plt.grid(True)
        plt.show()


class PointG:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def to_homogeneousp(self):
        coordinates2=np.array(self.coordinates)
        return np.array((coordinates2[0],coordinates2[1],1))
    
    def to_homogeneousp2(self):
        coordinates2=np.array(self.coordinates)
        return coordinates2/coordinates2[2]
    
    def list_points_to_homogeneous(lista_puntos):
        lista_puntos = lista_puntos
        puntos_homogeneos = []
        for punto in lista_puntos:
            puntos_homogeneos.append(punto.to_homogeneous())
        return puntos_homogeneos

    @staticmethod
    def build_from_lines(line1, line2):
        lineA = np.array(line1)
        lineB = np.array(line2)
        point_rest = np.cross(lineA, lineB)
        result = point_rest / point_rest[2]
        return PointG(result)


class LineG:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def to_homogeneousl(self):
        lineh = np.array(self.coefficients)
        return lineh/lineh[1]

    @staticmethod
    def build_from_points(point1, point2):
        pointA = np.array(point1)
        pointB = np.array(point2)
        line_rest = np.cross(pointA, pointB)
        return LineG(line_rest)

class ConicG:
    def __init__(self, C=None):
      self.C = C

    def to_homogeneousc(conica):
        arreglo=np.array(conica)
        a=arreglo[0]
        b=arreglo[1]
        c=arreglo[2]
        d=arreglo[3]
        e=arreglo[4]
        f=arreglo[5]
        C = np.array([[a, b/2, d/2],
                      [b/2, c, e/2],
                      [d/2, e/2, f]])
        return C


    def build_tan_point(self, point):
        if self.C is None:
            raise ValueError("La matriz de la cónica C no está definida")

        # Calcula la línea tangente a la cónica en el punto dado.
        punto=PointG((point))
        punto=punto.to_homogeneousp()
        matrix=self.C
        tangent_vector = np.dot(matrix,punto)
        tangent_line=LineG(tangent_vector)
        # Crea una instancia de la clase LineG con los coeficientes de la línea tangente.
        tangent_line = tangent_line.to_homogeneousl()


        return tangent_line

    def build_point_line(self, line):
        if self.C is None:
            raise ValueError("La matriz de la cónica C no está definida")

        # Calcula la línea tangente a la cónica en el punto dado.
        line=LineG((line))
        line=(line.to_homogeneousl())
        matrix=self.C
        punto = np.dot(np.linalg.inv(matrix),(line))
        # Crea una instancia de la clase PoinG con los coeficientes del punto.
        point_tangent_line=PointG(punto)
        point_tangent_line = point_tangent_line.to_homogeneousp2()
        return point_tangent_line
    
    def build_conic_points(points):
        points=np.array(points)
        # Construye la matriz A
        A = []
        for x, y in points:
            A.append([x**2, x*y, y**2, x, y, 1])
        A = np.array(A)
        # Construye el vector b (todo cero en este caso)
        b = np.zeros(A.shape[0])
        # Calcula la descomposición de valores singulares (SVD)
        U, S, Vt = np.linalg.svd(A)
        # Los coeficientes de la ecuación cónica están en la última fila de Vt
        coefficients = Vt[-1, :]
        # Los coeficientes están en el orden: [A, B, C, D, E, F]
        A, B, C, D, E, F = coefficients
        # Imprime los coeficientes
        conica_ecu=(A, B, C, D, E, F)
        return conica_ecu

class GeometryElement(ABC):
    @abstractmethod
    def plot(self, color):
        pass

class PointElement(GeometryElement):
    def __init__(self, point):
        self.point = point

    def plot(self, color):
        plt.plot(self.point[0], self.point[1], color)
        plt.text(self.point[0], self.point[1], f'({self.point[0]:.1f}, {self.point[1]:.1f})', fontsize=8, fontweight='bold', color='black')

    


class LineElement(GeometryElement):
    def __init__(self, line):
        self.line = line

    def plot(self, color):
        x_vals = np.linspace(-10, 10, 100)
        y_vals = ((-self.line[0] * x_vals) - self.line[2]) / self.line[1]
        plt.plot(x_vals, y_vals, color)

class ConicElement(GeometryElement):
    def __init__(self, conic_matrix):
        self.conic_matrix = conic_matrix

    def plot(self, color):
        # Definimos una matriz de transformación para deshomogeneizar las coordenadas
        # Normalizamos la matriz con la tercera coordenada de la forma general
        T = np.array([[1.0 / self.conic_matrix[2, 2], 0, -self.conic_matrix[0, 2] / self.conic_matrix[2, 2]],
                      [0, 1.0 / self.conic_matrix[2, 2], -self.conic_matrix[1, 2] / self.conic_matrix[2, 2]],
                      [0, 0, 1]])

        # Definimos una cuadrícula de puntos
        x_vals = np.linspace(-10, 10, 100)
        y_vals = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Evaluamos la ecuación de la cónica
        Z = self.conic_matrix[0, 0] * X**2 + 2 * self.conic_matrix[0, 1] * X * Y + self.conic_matrix[1, 1] * Y**2 + 2 * self.conic_matrix[0, 2] * X + 2 * self.conic_matrix[1, 2] * Y + self.conic_matrix[2, 2]

        # Aplicamos la transformación para deshomogeneizar las coordenadas
        transformed_points = np.dot(T, np.vstack([X.flatten(), Y.flatten(), np.ones_like(X.flatten())]))

        # Graficamos la cónica
        plt.contour(X, Y, Z, [0], colors=color)

        # Agregamos la ecuación de la cónica al gráfico
        equation_text = f'{self.conic_matrix[0, 0]:.3f}x^2 + {2 * self.conic_matrix[0, 1]:.3f}xy + {self.conic_matrix[1, 1]:.3f}y^2 + {2 * self.conic_matrix[0, 2]:.3f}x + {2 * self.conic_matrix[1, 2]:.3f}y + {self.conic_matrix[2, 2]:.3f} = 0'
        plt.text(-9, 9, equation_text, fontsize=8, color='m', fontweight='bold')