#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <cuda_runtime.h>
#include <cuda.h>

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class    
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return 100 * successfulPredictions / (float) dataset->num_instances();
}


__global__ void KNN(float *test_data, float *train_data, float *dist_list, int test_num_instances, int train_num_instances, int num_attributes)
{
	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	int tid = ( blockDim.x * gridDim.x * row ) + column;

	if (tid < train_num_instances * test_num_instances) {
		int testid = tid / train_num_instances;
		int trainid = tid % train_num_instances;
		float dist = 0;
        // loop through all the attributes, calculate the distance
		for (int i = 0; i < num_attributes - 1; ++i) {
            // approach 1:
            float diff = test_data[testid + i * test_num_instances] - train_data[trainid + i * train_num_instances];

            // approach 2:
			// float diff = test[testid * num_attributes + i] - train[trainid * num_attributes + i];
			dist += diff * diff;
		}
		dist_list[testid * train_num_instances + trainid] = dist;
	}

}

int main(int argc, char *argv[])
{
    if(argc != 5)
    {
        printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads_per_block");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);
    int thread_from_arg = strtol(argv[4], NULL, 10);


    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    int num_classes = train->num_classes();
    int train_num_attributes = train->num_attributes();
    int test_num_attributes = test->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    /*
    =====================================================================================
    Step 1: Re-arrange the train/test matrix, we will use num_test_instances × 
                                                num_train_instances threads
    // =====================================================================================
    */

    // approach 1:
    //================================================================================================

    float* h_train_matrix = (float*)malloc(train_num_attributes * train_num_instances * sizeof(float));
    float* h_test_matrix = (float*)malloc( test_num_attributes  * test_num_instances  * sizeof(float));
    
    for (int i = 0; i < train_num_instances; ++i){
        for(int feature_id = 0; feature_id < train_num_attributes; ++feature_id){
            h_train_matrix[i + feature_id * train_num_instances] = train->get_instance(i)->get(feature_id)->operator float();
        }
    }

    // printf("H Train Matrix:\n");

    // for (int i = 0; i < 1; ++i){
    //     for(int j = 0; j < train->num_attributes(); j++) {
    //         printf("%.2f ", h_train_matrix[i + j * train_num_instances]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < test_num_instances; ++i){
        for(int feature_id = 0; feature_id < test_num_attributes; ++feature_id){
            h_test_matrix[i + feature_id * test_num_instances] = test->get_instance(i)->get(feature_id)->operator float();
        }
    }

    //================================================================================================

    // approach 2:
    //================================================================================================
    /*
    float* h_train_matrix = (float*)malloc(train_num_attributes * train_num_instances * sizeof(float));
    float* h_test_matrix = (float*)malloc( test_num_attributes  * test_num_instances  * sizeof(float));
    
    for (int i = 0; i < train_num_instances; ++i){
        for(int feature_id = 0; feature_id < train_num_attributes; ++feature_id){
            h_train_matrix[i * train_num_attributes + feature_id] = train->get_instance(i)->get(feature_id)->operator float();
        }
    }

    printf("H Train Matrix:\n");

    for (int i = 0; i < 1; ++i){
        for(int j = 0; j < train->num_attributes(); j++) {
            printf("%.2f ", h_train_matrix[i * train_num_instances + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < test_num_instances; ++i){
        for(int feature_id = 0; feature_id < test_num_attributes; ++feature_id){
            h_test_matrix[i * test_num_attributes + feature_id] = test->get_instance(i)->get(feature_id)->operator float();
        }
    }
    */
    //================================================================================================

    // Step 2: Allocate the device input 

    float* h_dist      = (float*)malloc(test_num_instances * train_num_instances * sizeof(float));
    int* predictions = (int*)malloc(test_num_instances * sizeof(int));

    float*d_train, *d_test, *d_dist;

    cudaMalloc(&d_dist, train_num_instances * test_num_instances * sizeof(float));
    cudaMalloc(&d_train, train_num_attributes * train_num_instances * sizeof(float));
    cudaMalloc(&d_test, test_num_attributes * test_num_instances * sizeof(float));

    // Step 3:  Copy the host input h_train_matrix and h_test_matrix in host memory to the device input 
    cudaMemcpy(d_train, h_train_matrix, train_num_attributes * train_num_instances * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test_matrix, test_num_attributes * test_num_instances * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
	int blocksPerGrid = (train_num_instances * test_num_instances + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

    cudaEventRecord(start);
	KNN<<<blocksPerGrid, threadsPerBlock>>>(d_test, d_train, d_dist, test_num_instances, train_num_instances, train_num_attributes);

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to find the distance: %f ms\n", milliseconds);

    // Transfer from GPU memory to CPU memory.
    cudaMemcpy(h_dist, d_dist, train_num_instances * test_num_instances * sizeof(float), cudaMemcpyDeviceToHost);


    struct timespec start_cpu, end_cpu;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start_cpu);

    for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float* candidates = (float*) calloc(k*2, sizeof(float));
        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

        // Stores bincounts of each class over the final set of candidate NN. Calloc initializes values with 0s
        int* classCounts = (int*)calloc(num_classes, sizeof(int));

        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            float dist = h_dist[queryIndex * train_num_instances + keyIndex];
            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float(); // class value

                    break;
                }
            }
            // printf("Thread: %d, Query Index: %d, key index: %d, dist %f \n", omp_get_num_threads(), queryIndex, keyIndex, dist);
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2*i+1]] += 1;
        }

        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2*i+1]] += 1;
        }
        
        int max_value = -1;
        int max_class = 0;
        for(int i = 0; i < num_classes; i++) {
            if(classCounts[i] > max_value) {
                max_value = classCounts[i];
                max_class = i;
            }
        }

        // Make prediction with 
        predictions[queryIndex] = max_class;
        
        free(candidates);
        free(classCounts);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_cpu);

    // For debugging purposes, print the distance matrix
    // for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
    //     for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
    //         printf("%d ", h_dist[queryIndex * train_num_instances + keyIndex]);
    //     }
    //     printf("\n");
    // }

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    double time_difference = (1000000000 * (end_cpu.tv_sec - start_cpu.tv_sec) + end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e6;
    printf("CPU time to find the distance: %f ms\n", time_difference);
    time_difference += milliseconds; // + with the GPU time

    printf("The %i-NN classifier for %lu test instances and %lu train instances required %.2f ms CPU time for single-thread. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), time_difference, accuracy);


    free(predictions);
    free(confusionMatrix);

    cudaFree(d_dist);
    cudaFree(d_train);
    cudaFree(d_test);

    free(h_train_matrix);
    free(h_test_matrix);
    free(h_dist);
    /*
    // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
    */
}

