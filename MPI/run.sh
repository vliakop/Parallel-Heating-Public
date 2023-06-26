clear
rm -i ./mpimachines
./askdate_mpi.csh
rm -i ./openmpmachines
./askdate_openmp.csh
#Make executable for MPI
clear
make clean
make
echo "--------------->MPI<---------------" > ./results.txt
#Run executable for every given number o processes
for i in "$@";
do
         echo "--> $i process -----------------------" | tee -a results.txt
         echo "" | tee -a results.txt
         echo "MPI - $i process"
         mpiexec -f mpimachines -n $i ./heatMap | tee -a results.txt
         echo "-------------------------------------------------------" | tee -a results.txt

done  

echo "*********************************************************************" | tee -a results.txt
echo "" | tee -a results.txt

#Make executable for OPENMP
make clean
make OPENMP=1
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
