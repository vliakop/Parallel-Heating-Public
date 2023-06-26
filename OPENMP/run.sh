echo "------------->OPENMP<-------------" | tee -a results.txt
#Run executable for every given number o processes
for i in "$@";
do
         echo "--> $i process -----------------------" | tee -a results.txt
         echo "" | tee -a results.txt
         echo "OPENMP - $i process"
         mpiexec -f openmpmachines -n $i ./heatMap | tee -a results.txt
         echo "-------------------------------------------------------" | tee -a results.txt
done
