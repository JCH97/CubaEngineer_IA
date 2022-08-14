Para la resolución del problema se emplea Regresión Logística.

Se tienen 4 posibles clasificaciones para los números:
1 => Fizz => 3 | n
2 => Buzz => 5 | n
3 => FizzBuzz => 15 | n
4 => None

En la tercera, que es para comprabar si el número divide tanto a 3 como a 5, basta con revisar si 15 | n (15 divide a n)
, puesto que como 3 y 5 son primos (y se tendría como hipótesis que 3 | n y 5 | n) entonces el producto de ellos también
divide a n.

La idea detrás de la solución del problema radica en tomar como features la representación en binario de los números,
para luego predecir a qué clase pertenecería el número.

Para crear el dataset se están usando 10 bits, es decir, todos los números con 10 bits, serian 2 ** 10 números. De ellos
el 33% para testing y el resto para entrenamiento.

Se probó con un mayor número de bits, para garantizar que el tamaño del dataset fuera mayor y ganar en valores a la hora
de entrenar el modelo, pero los mejores resultados se obtuvieron con los parámetros con los que se entrega código. Se
decide usar 10 bits, pues si se aumenta dicha cantidad aparecen muchos más números que no son divisibles ni por 3 ni por
5 (por consiguiente por 15 tampoco), luego el modelo estaba haciendo overfit y solo aprendiendo la forma de estos tipos
de números (los None), entonces las predicciones no eran tan buenas; en este momento se tiene un modelo con un 54% de
probabilidades de acierto.

Una buena estrategia podría consistir quizá en aplicar un algoritmo de reducción de dimension (PCA, ICA) para tener una
representación (quizá en dos dimensiones) del problema y con ello poder visualizar mejor los datos (graficarlos) para
tener asi una idea más clara de como están distribuidos y saber ajustar los parámetros para ganar en aciertos.

Se realizó 10-fold-validation con el mismo algoritmo de clasificación (se obtienen 10 modelos diferentes) para observar
como el modelo se comporta ante nuevos datos, los resultados se pueden ver en la ejecución del código.

En general, considero que quizá usando un algoritmo de DL, se pudiera mejorar mucho este problema y obtener los mejores
resultados, o quizá los resultados esperados.

Quizá por falta de tiempo, no pude adjuntar más pruebas ni llegar a un análisis más concreto de los datos. Más adelante
enviaré la solución al problema con DL para comparar los resultados.