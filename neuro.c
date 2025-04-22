#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "errors.c"


#ifndef NEURAL_NETWORK_LIB
#define NEURAL_NETWORK_LIB

#define type double


type randValue(){
    static char randomizer_created = 0;
    if (!randomizer_created){
        randomizer_created = 1;
        srand((time(NULL) << 11) + 127);
    }
    return 2 * ((type)rand() / (type)RAND_MAX) - 1;
}

void checkNULL_IN_NEURAL_NETWORK_LIB(void* ptr){
    if (!ptr){
        printf("BAD MEMORY ALLOCATION!\n");
        exit(1);
    }
}

// typedef int (*activationFunction)(type);

typedef struct Neuron {
    type value;
    type before_activation;
    type bias;
    type* weights;
} Neuron;

typedef type (*function)(type);

typedef struct Layer {
    size_t len;
    function func;
    function derivate_func;
    Neuron* neurons;
} Layer;

Layer* createLayer(size_t length, function func, function derivate_func, size_t weightsCount){
    Layer* result = (Layer*)malloc(sizeof(Layer));
    checkNULL_IN_NEURAL_NETWORK_LIB(result);
    result->len = length;
    result->func = func;
    result->derivate_func = derivate_func;
    result->neurons = (Neuron*)calloc(length, sizeof(Neuron));
    checkNULL_IN_NEURAL_NETWORK_LIB(result->neurons);
    if (weightsCount) {
        for (size_t i = 0; i < length; i++){
            result->neurons[i].weights = (type*)malloc(weightsCount * sizeof(type));
            checkNULL_IN_NEURAL_NETWORK_LIB(result->neurons[i].weights);
            result->neurons[i].bias = randValue();      //(type)0;
            for (size_t j = 0; j < weightsCount; j++){
                result->neurons[i].weights[j] = randValue();
            }
        }
    }
    return result;
}

Layer* createInputLayer(size_t length){
    return createLayer(length, NULL, NULL, 0);
}

int destroyLayer(Layer* layer){
    if (!layer) return 0;
    for (size_t i = 0; i < layer->len; i++){
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer);
    return 0;
}

void printLayer(Layer* layer){
    for (size_t i = 0; i < layer->len; i++){
        printf("%.2f%s", layer->neurons[i].value, i + 1 == layer->len ? "" : " ");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Activate Functions /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

type sigmoid(type value){
    return (type)1 / (type)(1 + exp(-value));
}

type derivativeSigmoid(type value){
    return sigmoid(value) * ((type)1 - sigmoid(value));
}

type getAlphaForELU(){
    type alpha = 1.0;
    return alpha;
}

type ELU(type value){
    type alpha = getAlphaForELU();
    if (value > 0) return value;
    return (type)(alpha * (exp(value) - 1));
}

type derivativeELU(type value){
    type alpha = getAlphaForELU();
    if (value > 0) return (type)1.0;
    return (type)(alpha * exp(value));
}

type getAlphaForLeakyReLU(){
    type alpha = 0.01;
    return alpha;
}

type LeakyReLU(type value){
    type compare = value * getAlphaForLeakyReLU();
    if (value > compare) return value;
    return compare;
}

type derivativeLeakyReLU(type value){
    type alpha = getAlphaForLeakyReLU();
    if (value > 0) return (type)(1);
    return alpha;
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

typedef struct NeuralNetwork {
    size_t countLayers;
    Layer** layers;
} NeuralNetwork;

NeuralNetwork* createNeuralNetwork(size_t* sizes, char* funcName){
    function func;
    function derivate_func;
    
    if (!strcmp(funcName, "sigmoid")){
        func = sigmoid;
        derivate_func = derivativeSigmoid;
    } else if (!strcmp(funcName, "leakyrelu")){
        func = LeakyReLU;
        derivate_func = derivativeLeakyReLU;
    } else if (!strcmp(funcName, "elu")){
        func = ELU;
        derivate_func = derivativeELU;
    } else {
        printf("BAD CHOOSE ACTIVATE FUNCTION");
        exit(1);
    }

    size_t count = sizes[0];
    if (!count) return NULL;
    NeuralNetwork* result = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    checkNULL_IN_NEURAL_NETWORK_LIB(result);
    result->countLayers = count;
    result->layers = (Layer**)calloc(count, sizeof(Layer*));
    checkNULL_IN_NEURAL_NETWORK_LIB(result->layers);
    for (size_t i = 0; i < count; i++){
        if (!i)
            result->layers[i] = createInputLayer(sizes[i + 1]);
        else
            result->layers[i] = createLayer(sizes[i + 1], func, derivate_func, sizes[i]);
    }
    return result;
}

int destroyNeuralNetwork(NeuralNetwork* nn){
    if (!nn) return 0;
    for (size_t i = 0; i < nn->countLayers; i++){
        destroyLayer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
    return 0;
}

int setInputValues(type* values, NeuralNetwork* nn){
    for (size_t i = 0; i < nn->layers[0]->len; i++){
        nn->layers[0]->neurons[i].value = values[i];
    }
    return 0;
}

int getOutputValues(type* values, NeuralNetwork* nn){
    size_t index = nn->countLayers - 1;
    for (size_t i = 0; i < nn->layers[index]->len; i++){
        values[i] = nn->layers[index]->neurons[i].value;
    }
    return 0;
}

int getOutputValuesSoftMax(type* values, NeuralNetwork* nn){
    size_t index = nn->countLayers - 1;
    type absSum = 0;
    for (size_t i = 0; i < nn->layers[index]->len; i++){
        absSum += exp(nn->layers[index]->neurons[i].value);
    }
    for (size_t i = 0; i < nn->layers[index]->len; i++){
        values[i] = (type)(exp(nn->layers[index]->neurons[i].value) / absSum);
    }
    return 0;
}

size_t getLengthInputLayer(NeuralNetwork* nn){
    return nn->layers[0]->len;
}

size_t getLengthOutputLayer(NeuralNetwork* nn){
    return nn->layers[nn->countLayers - 1]->len;
}

void forwardPropagation(NeuralNetwork* nn){
    if (nn->countLayers < 1){
        raiseError(lenerr);
    }
    Layer* preLayer;
    Layer* layer;
    Layer** layers = nn->layers;
    for (size_t i = 1; i < nn->countLayers; i++){
        preLayer = layers[i - 1];
        layer = layers[i];
        if ((layer->len < 1) || (preLayer->len < 1)){
            raiseError(lenerr);
        }
        for (size_t j = 0; j < layer->len; j++){
            type resultValue = 0;
            for (size_t k = 0; k < preLayer->len; k++){
                resultValue += preLayer->neurons[k].value * layer->neurons[j].weights[k];
            }
            resultValue += layer->neurons[j].bias;

            layer->neurons[j].value = layer->func(resultValue);
            layer->neurons[j].before_activation = resultValue;
        }
    }
    return;
}

void backwardPropagation(NeuralNetwork* nn, type* targets, type learningRate) {
    if (nn->countLayers < 2) {
        raiseError(lenerr);
        return;
    }

    Layer* outputLayer = nn->layers[nn->countLayers - 1];
    size_t outputLen = outputLayer->len;
    
    type* deltas = (type*)malloc(outputLen * sizeof(type));

    for (size_t i = 0; i < outputLen; i++) {
        type output = outputLayer->neurons[i].value;
        type before_act = outputLayer->neurons[i].before_activation;
        type error = output - targets[i];
        deltas[i] = error * outputLayer->derivate_func(before_act);
    }

    for (int layerIdx = nn->countLayers - 1; layerIdx >= 1; layerIdx--) {
        Layer* currentLayer = nn->layers[layerIdx];
        Layer* prevLayer = nn->layers[layerIdx - 1];

        type* newDeltas = NULL;
        if (layerIdx > 1) {
            newDeltas = (type*)malloc(prevLayer->len * sizeof(type));
        }

        for (size_t neuronIdx = 0; neuronIdx < currentLayer->len; neuronIdx++) {
            Neuron* neuron = &currentLayer->neurons[neuronIdx];

            for (size_t prevNeuronIdx = 0; prevNeuronIdx < prevLayer->len; prevNeuronIdx++) {
                neuron->weights[prevNeuronIdx] -= learningRate * deltas[neuronIdx] 
                                                * prevLayer->neurons[prevNeuronIdx].value;
            }

            neuron->bias -= learningRate * deltas[neuronIdx];

            if (layerIdx > 1) {
                for (size_t prevNeuronIdx = 0; prevNeuronIdx < prevLayer->len; prevNeuronIdx++) {
                    newDeltas[prevNeuronIdx] += currentLayer->neurons[neuronIdx].weights[prevNeuronIdx] 
                                              * deltas[neuronIdx];
                }
            }
        }

        if (layerIdx > 1) {
            for (size_t prevNeuronIdx = 0; prevNeuronIdx < prevLayer->len; prevNeuronIdx++) {
                newDeltas[prevNeuronIdx] *= prevLayer->derivate_func(
                    prevLayer->neurons[prevNeuronIdx].before_activation
                );
            }
        }

        free(deltas);
        deltas = newDeltas;
    }
    free(deltas);
}

void printNeuralNetwork(NeuralNetwork* nn){
    for (size_t i = 0; i < nn->countLayers; i++){
        printLayer(nn->layers[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
int putc8bytes(void* bytes, FILE* file){
    fwrite((char*)bytes, sizeof(char), 8, file);
    return 0;
}

int putc4bytes(void* bytes, FILE* file){
    fwrite((char*)bytes, sizeof(char), 4, file);
    return 0;
}

int saveNeuralNetwork(char* filename, NeuralNetwork* nn){
    FILE* file = fopen(filename, "wb");
    if (!file){
        printf("FILE OPEN ERROR!\n");
        fclose(file);
        exit(1);
    }
    putc8bytes(nn, file);
    size_t lenPreLayer = 0;
    for (size_t i = 0; i < nn->countLayers; i++){
        putc8bytes((char*)nn->layers[i], file);
    }

    for (size_t i = 0; i < nn->countLayers; i++){
        if (!i) continue;
        lenPreLayer = nn->layers[i - 1]->len;
        Neuron* neurons = nn->layers[i]->neurons;
        for (size_t j = 0; j < nn->layers[i]->len; j++){
            fwrite(&(neurons[j].bias), sizeof(type), 1, file);
            fwrite(neurons[j].weights, sizeof(type), lenPreLayer, file);
        }
    }
    fclose(file);
    return 0;
}

NeuralNetwork* loadNeuralNetwork(char* filename){
    FILE* file = fopen(filename, "rb");
    if (!file){
        printf("FILE OPEN ERROR!\n");
        exit(1);
    }
    size_t countLayers = 0;
    if (fread(&countLayers, sizeof(size_t), 1, file) != 1){
        printf("FILE READ ERROR 1!\n");
        exit(1);
    }
    size_t* sizes = (size_t*)malloc((countLayers + 1) * sizeof(size_t));
    checkNULL_IN_NEURAL_NETWORK_LIB(sizes);
    sizes[0] = countLayers;
    fread(&sizes[1], sizeof(size_t), countLayers, file);

    NeuralNetwork* resultNN = createNeuralNetwork(sizes, "sigmoid");
    for (size_t i = 1; i < countLayers; i++){
        for (size_t j = 0; j < sizes[i + 1]; j++){
            int status1 = fread(&resultNN->layers[i]->neurons[j].bias, sizeof(type), 1, file);
            int status2 = fread((resultNN->layers[i]->neurons[j].weights), sizeof(type), sizes[i], file);
            if (status1 != 1){
                printf("FILE READ ERROR status1!\n");
                free(sizes);
                exit(1);
            }
            if (status2 != sizes[i]){
                printf("FILE READ ERROR status2!\n");
                free(sizes);
                exit(1);
            }
        }
    }
    free(sizes);
    fclose(file);
    return resultNN;
}

/*
int my_fwrite(void* string, FILE* file){
    char* _string = (char*)string;
    fwrite(_string, sizeof(size_t), 1, file);
}

int test_fwrite(){
    FILE* file = fopen("testing3.nn", "wb");
    size_t someInformation[] = {4};
    my_fwrite(someInformation, file);
    return 0;
}*/


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

/*
int main(){
    size_t sizes[] = {3, 2, 2, 2};
    NeuralNetwork* nn = createNeuralNetwork(sizes, "sigmoid");

    forwardPropagation(nn);
    printNeuralNetwork(nn);
    return 0;
}*/

/*
int test1(){
    size_t layers[] = {4, 2, 2, 2, 2};
    NeuralNetwork* nn = createNeuralNetwork(layers, "sigmoid");
    type values[] = {0.2, 0.8};
    setInputValues(values, nn);
    forwardPropagation(nn);
    printNeuralNetwork(nn);

    type y[] = {0.5, 0.6};
    for (int i = 0; i < 1000; i++){
        backwardPropagation(nn, y, 0.005);
        forwardPropagation(nn);
    }
    printf("\n");
    printNeuralNetwork(nn);
    saveNeuralNetwork("testing.nn", nn);
}

int test2(){
    NeuralNetwork* nn = loadNeuralNetwork("testing4.nn");
    type values[] = {0.2, 0.8};
    setInputValues(values, nn);
    forwardPropagation(nn);
    printNeuralNetwork(nn);
}

int main(){
    test2();
    return 0;
}
*/
#endif



