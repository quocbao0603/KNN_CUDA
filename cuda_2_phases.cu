#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
using namespace std;

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
		dist_list[testid * train_num_instances + trainid] = sqrt(dist);
	}
}

__global__ void find_k_nearest(int* predictions, float* train_class, float* d_dist, int test_num_instances, int train_num_instances, 
                                int* classCounts, float* candidates, int num_classes, int max_value, int max_class, int k)
{
    int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	int tid = ( blockDim.x * gridDim.x * row ) + column;
    // printf("%d %d\n", tid, test_num_instances);
    if (tid < test_num_instances) {
        int testid = tid;
        // printf("testing %d:\n", testid);
        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        for(int i = 0; i < 2*k; i++){ candidates[testid * k * 2 + i] = FLT_MAX; }
        // if (tid == 0){
        //     printf("TESTING:\n");
        //     for(int i = 0; i < train_num_instances; ++i){
        //         printf("%f ", train_class[i]);
        //     }
        //     printf("\n");
        // }
        for(int trainid = 0; trainid < train_num_instances; trainid++) {
            float dist = d_dist[testid * train_num_instances + trainid];
            // Add to our candidates
            // if (dist > 0){
            // if (train_class[trainid] > 0)
                // printf("Query Index: %d, key index: %d, k = %d, dist %f, class: %d \n", testid, trainid, k, dist, train_class[trainid]);
            // }
            for(int c = 0; c < k; c++){
                if(dist < candidates[testid * k * 2 + 2*c]) {
                    // printf("Found a new candidate:\n");
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[testid * k * 2 + 2*x+2] = candidates[testid * k * 2 + 2*x];
                        candidates[testid * k * 2 + 2*x+3] = candidates[testid * k * 2 + 2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[testid * k * 2 + 2*c] = dist;
                    candidates[testid * k * 2 + 2*c+1] = train_class[trainid];
                    // candidates[2*c+1] = train->get_instance(trainid)->get(train->num_attributes() - 1)->operator float(); // class value

                    break;
                }
            }
        }
        // printf("Done loop train: %d", testid);

        for(int i = 0; i < k; i++) {
            classCounts[num_classes * testid + (int)candidates[testid * k * 2 + 2*i+1]] += 1;
        }

        // for (int i = 0; i < k; i++){
        //     if (i == 0){
        //         printf("testid: %d ", testid);
        //     }
        //     printf("%f %f ", candidates[2*i], candidates[2*i + 1]);
        //     if (i == k - 1){
        //         printf("\n");
        //     }
        // }
        
        
        predictions[testid] = 0;
        for(int i = 0; i < num_classes; i++) {
            if(classCounts[num_classes * testid + i] > classCounts[num_classes * testid + predictions[testid]]) {
                predictions[testid] = i;
                // printf("update\n");
            }
        }
        
        // Make prediction with 
        // predictions[testid] = max_class;
        // printf("%d %d\n", testid, predictions[testid]);
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

    // =============== begin of phase 1: calculate the distance  ==================
    printf("===============Start phase 1================\n");
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
    int* h_predictions = (int*)malloc(test_num_instances * sizeof(int));

    float*d_train, *d_test, *d_dist;

    cudaMalloc(&d_dist, train_num_instances * test_num_instances * sizeof(float));
    cudaMalloc(&d_train, train_num_attributes * train_num_instances * sizeof(float));
    cudaMalloc(&d_test, test_num_attributes * test_num_instances * sizeof(float));

    // Step 3:  Copy the host input h_train_matrix and h_test_matrix in host memory to the device input 
    cudaMemcpy(d_train, h_train_matrix, train_num_attributes * train_num_instances * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test_matrix, test_num_attributes * test_num_instances * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = thread_from_arg;
	int blocksPerGrid = (train_num_instances * test_num_instances + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

    cudaEventRecord(start);
	KNN<<<blocksPerGrid, threadsPerBlock>>>(d_test, d_train, d_dist, test_num_instances, train_num_instances, train_num_attributes);

    // Transfer from GPU memory to CPU memory.
    cudaMemcpy(h_dist, d_dist, train_num_instances * test_num_instances * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 10; ++i){
    //     printf("%f ", h_dist[i]);
    // }
    // printf("\n");
    printf("===============Done phase 1================\n");
    // =============== end of phase 1 ==================

    // cudaDeviceSynchronize();

    // =============== begin of phase 2: find the k-nearest neighbors ==================
    printf("===============Start phase 2================\n");
    
    int* d_predictions, *d_classCounts;
    float* d_candidates, *d_dist_phase_2, *d_train_class;
    
    // copy train dataset
    float* h_train_class      = (float*)malloc(train_num_instances * sizeof(float));
    for (int i = 0; i < train_num_instances; ++i){
        h_train_class[i] = train->get_instance(i)->get(train_num_attributes - 1)->operator float(); // class value
        // if (i == 157)
        //     printf("i: %d, class: h_train_class: %f", i, h_train_class[i]);
    }
    
    cudaMalloc(&d_predictions, test_num_instances * sizeof(int));
    cudaMalloc(&d_dist_phase_2, train_num_instances * test_num_instances * sizeof(float));
    cudaMalloc(&d_train_class, train_num_instances * sizeof(float));
    cudaMalloc(&d_candidates, k * 2 * test_num_instances * sizeof(float));
    cudaMalloc(&d_classCounts, num_classes * test_num_instances);

    cudaMemcpy(d_dist_phase_2, h_dist, train_num_instances * test_num_instances * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_class, h_train_class, train_num_instances * sizeof(float), cudaMemcpyHostToDevice);
    int max_class = 0, max_value = -1;

    blocksPerGrid = (test_num_instances + threadsPerBlock - 1) / threadsPerBlock;

    find_k_nearest<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_train_class, d_dist_phase_2, test_num_instances, train_num_instances, d_classCounts, d_candidates, num_classes, max_value, max_class, k);
    
    cudaMemcpy(h_predictions, d_predictions, test_num_instances * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

    printf("===============Done phase 2================\n");
	printf("GPU time to find the distance: %f ms\n", milliseconds);

    // for (int i = 0; i < test_num_instances; ++i){
    //     printf("%d %d\n", i, h_predictions[i]);
    // }

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(h_predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms GPU time for multiple threads. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) milliseconds, accuracy);

    free(h_predictions);
    free(confusionMatrix);
    
    cudaFree(d_dist);
    cudaFree(d_train);
    cudaFree(d_test);
    cudaFree(d_predictions);
    cudaFree(d_classCounts);
    cudaFree(d_candidates);
    cudaFree(d_train_class);
    cudaFree(d_dist_phase_2);

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

