#!/bin/bash
make clean && make

SIZES=(1 2 3 4)
CPUS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
for size in ${SIZES[@]}; do
	for cpu in ${CPUS[@]}; do
		pt1="output/mpi_"
		pt2="size$size"
		pt3="_N1_CPU_$cpu.out"
		string=$pt1$pt2$pt3
		if [ $size -eq "1" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu mpi_128_128.job
		elif [ $size -eq "2" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu mpi_128_256.job
		elif [ $size -eq "3" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu mpi_256_256.job
		elif [ $size -eq "4" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu mpi_1024_1024.job
		fi
	done
done

SIZES=(1 2 3 4)
CPUS=(14 15 16 17 18 19 20 21 22 23 24 25 26 27 28)
for size in ${SIZES[@]}; do
	for cpu in ${CPUS[@]}; do
		pt1="output/mpi_"
		pt2="size$size"
		pt3="_N1_CPU_$cpu.out"
		string=$pt1$pt2$pt3
		if [ $size -eq "1" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu --partition cpu mpi_128_128.job
		elif [ $size -eq "2" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu --partition cpu mpi_128_256.job
		elif [ $size -eq "3" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu --partition cpu mpi_256_256.job
		elif [ $size -eq "4" ]; then
			sbatch --output $string --nodes 1 --ntasks-per-node $cpu --partition cpu mpi_1024_1024.job
		fi
	done
done

SIZES=(1 2 3 4)
NODES=(1 2 3 4)
for size in ${SIZES[@]}; do
	for node in ${NODES[@]}; do
		pt1="output/mpi_"
		pt2="size$size"
		pt3="_N$node"
        pt4="_CPU_28.out"
		string=$pt1$pt2$pt3$pt4
		if [ $size -eq "1" ]; then
			sbatch --output $string --nodes $node --ntasks-per-node 28 --partition cpu mpi_128_128.job
		elif [ $size -eq "2" ]; then
			sbatch --output $string --nodes $node --ntasks-per-node 28 --partition cpu mpi_128_256.job
		elif [ $size -eq "3" ]; then
			sbatch --output $string --nodes $node --ntasks-per-node 28 --partition cpu mpi_256_256.job
		elif [ $size -eq "4" ]; then
			sbatch --output $string --nodes $node --ntasks-per-node 28 --partition cpu mpi_1024_1024.job
		fi
	done
done
