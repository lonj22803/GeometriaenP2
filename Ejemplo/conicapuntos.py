import sys
sys.path.append("./") # Adds higher directory to python modules path.

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from Geometry.geometryp2 import HomogeneousGeometry, PointG, LineG, ConicG, GeometryElement,PointElement, LineElement, ConicElement
from Seleccion.selectpoints import SeleccionDePuntos


#Corramos la clase Seleccionador de puntos
seleccionador = SeleccionDePuntos(num_puntos_minimos=5)
seleccionador.show()
#obtengamos los puntos seleccionados en una lista
puntos= list(seleccionador.points)
#Usemos la Clase conica para contuir la ecuacion de la conica a partir de los puntos
ecu_conica= ConicG.build_conic_points(puntos)
#Utilizamos un objeto conica para contruirlo a partir de la ecuacion.
conica= ConicG()
conic_mtrix= ConicG.to_homogeneousc(ecu_conica)
conica.C=conic_mtrix
#Grafiquemos la conica y los puntos
grafica = HomogeneousGeometry()
grafica.add_element(ConicElement(conica.C), "b")
for punto in puntos:
    punto=PointG(punto)
    grafica.add_element(PointElement(punto.to_homogeneousp()), "ro")
grafica.plot()