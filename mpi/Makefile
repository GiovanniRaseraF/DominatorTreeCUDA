hello:
	mpic++ hello.cpp -o hello.out
	mpirun --oversubscribe -np 10 ./hello.out

send:
	mpic++ send.cpp -o send.out
	mpirun --oversubscribe -np 4 ./send.out

deadlock:
	mpic++ d.cpp -o d.out
	mpirun --oversubscribe -np 4 ./d.out

circular:
	mpic++ circular.cpp -o circular.out
	mpirun --oversubscribe -np 4 ./circular.out

reduce:
	mpic++ reduce.cpp -o reduce.out
	mpirun --oversubscribe -np 5 ./reduce.out

allreduce:
	mpic++ allreduce.cpp -o allreduce.out
	mpirun --oversubscribe -np 5 ./allreduce.out

scan:
	mpic++ scan.cpp -o scan.out
	mpirun --oversubscribe -np 3 ./scan.out



