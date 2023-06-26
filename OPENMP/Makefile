MPICXX = mpicc
OUTPUT = heatMap
OBJECTS = heat.o
EXEC = heatMap
FLAGS = -lm


heat.o: heat.c 
	$(MPICXX) -c -g -Wall -fopenmp ompi_heat.c $(FLAGS)
                         
clean: 
		rm -rf $(EXEC) *.o *~ $(OUTPUT)            
