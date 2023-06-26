MPICXX = mpicc
OUTPUT = heatMap
OBJECTS = heat.o
EXEC = heatMap
FLAGS = -lm


heat.o: heat_vectors.c 
	$(MPICXX) -c -g -Wall heat_subArray.c $(FLAGS)
                         
clean: 
		rm -rf $(EXEC) *.o *~ $(OUTPUT)            
