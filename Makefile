all: serial threaded openmp cuda cuda_2_phases assignment2_test 
serial: serial.cpp
	g++ -std=c++11 -o serial serial.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
threaded: threaded.cpp
	g++ -pthread -std=c++11 -o threaded threaded.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
openmp: openmp.cpp
	g++ -fopenmp -std=c++11 -o openmp openmp.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
mpi: mpi.cpp
	mpicxx -std=gnu++11 -o mpi mpi.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda: 
	nvcc --gpu-architecture=compute_90 --gpu-code=sm_90 -std=c++11 -o cuda cuda.cu -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp 
cuda_2_phases: 
	nvcc --gpu-architecture=compute_90 --gpu-code=sm_90 -std=c++11 -o cuda_2_phases cuda_2_phases.cu -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp 
assignment2_test: 
	nvcc -std=c++11 -o assignment2_test assignment2_test.cu -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp 
clean:
	rm serial threaded openmp cuda cuda_2_phases assignment2_test 