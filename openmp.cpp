#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
// #include <vector>
// Calculates the distance between two instances
float distance(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    
    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff*diff;
    }
    
    return sqrt(sum);
}

// Implements a OpenMP kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int* KNN(ArffData* train, ArffData* test, int k, int max_num_threads) {    
    
    /************************STARTING MY IMPLEMENTATION*****************************/
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int* predictions = (int*)malloc(test_num_instances * sizeof(int));

    

    // Pointers representing the dataset as a 2D matrix size num_instances x num_attributes
    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();

    #pragma omp parallel 
    #pragma omp for
    for(int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {
        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float* candidates = (float*) calloc(k*2, sizeof(float));
        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

        // Stores bincounts of each class over the final set of candidate NN. Calloc initializes values with 0s
        int* classCounts = (int*)calloc(num_classes, sizeof(int));

        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
            // if (queryIndex == 159)printf("Thread: %d, Query Index: %d, key index: %d, train: %f, test %f \n", omp_get_num_threads(), queryIndex, keyIndex, test_matrix[queryIndex*num_attributes],  train_matrix[keyIndex*num_attributes]);
            float dist = distance(&test_matrix[queryIndex*num_attributes], &train_matrix[keyIndex*num_attributes], num_attributes);
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
                    candidates[2*c+1] = train_matrix[keyIndex*num_attributes + num_attributes - 1]; // class value

                    break;
                }
            }
            // printf("Thread: %d, Query Index: %d, key index: %d, dist %f \n", omp_get_num_threads(), queryIndex, keyIndex, dist);
        }

        // Bincount the candidate labels and pick the most common
        // printf("Query Index: %d\n", queryIndex);
        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2*i+1]] += 1;
        //     printf("%d ", classCounts[(int)candidates[2*i+1]]);
        }
        // printf("\n");
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

        // // Reset arrays
        // for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        // memset(classCounts, 0, num_classes * sizeof(int));
        free(candidates);
        free(classCounts);
    }

    /*****************************ENDING MY IMPLEMENTATION**************************/

    return predictions;
}

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

int main(int argc, char *argv[])
{
    if(argc != 5)
    {    
        printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads");
        exit(0);
    }

    // k value for the k-nearest neighbors
    int k = strtol(argv[3], NULL, 10);
    int num_threads = strtol(argv[4], NULL, 10);

    omp_set_num_threads(num_threads);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();
    
    struct timespec start, end;
    int* predictions = NULL;
    
    // Initialize time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    predictions = KNN(train, test, k, num_threads);

    // Stop time measurement
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("OpenMP num_thread: %d\n", num_threads);
    printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for OpenMP. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) time_difference, accuracy);
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/