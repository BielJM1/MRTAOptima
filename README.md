# MRTAOptima
En este repositorio presentamos el código utilizado para generar las simulaciones analizadas en el artículo: "Una aplicación de conjuntos difusos a la asignación óptima de tareas"

Con este código implementamos un nuevo método para la asignación óptima de tareas en sistemas multi-robot, capaz de trabajar con tareas que tienen asociados plazos límite de tiempo (time deadlines). Para esta MRTA nos inspiramos en los métodos de umbral de respuesta, pero modelando los estímulos percibidos por los robots a través de conjuntos difusos. Para tomar la decisión sobre cuál es la mejor tarea a realizar en cada momento, cada robot soluciona un problema de optimización difuso mediante la célebre técnica de Bellman y Zadeh.

Presentamos en el siguiente pseudocódigo, una idea de la estructura del código que se puede encontrar en el fichero "MAgentsOpt_Method1U2.py":
```python
{INITIALIZATIONS}
createTasks(m)
createRobots(n)
t<-0

{MAIN LOOP} 
while all tasks are not completed do
    for each robot a_k do
        bestTask <- None
        bestStimulus <- 0
        for each task t_j with t_j_et > 0 do
            if sk_cl_j_t > bestStimulus then
                bestTask <- t_j
                bestStimulus <- sk_cl_j_t
            end if
        end for

        if bestTask is not None then
            if robot a_k is not located at bestTask then
                ak.goTo(bestTask)
            else
                ak.workOn(bestTask)
            endif
        else
            ak.stop()
        end if
    end for
    t <- t+1
end while

{EFECTIVENESS ASSESSMENT}
computeMetrics()
````

Todo el código fuente ha sido diseñado utilizando el lenguaje Python 3.
