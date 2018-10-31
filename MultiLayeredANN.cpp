#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace std;

/*
* Xhanti Singatha
* SNGXHA002 
* MLLab 6
*/

double sigmoid(double value);
double sumWeights(int x1, vector<double> input);
double primeSigmoid(double net);

double sigmoid(double value){
       double ans = 1 / ( 1 + exp(-value));
       return ans;
}

double sumWeights(double x1, double x2, vector<double> input){
       double sum = (x1 * input[0]) + (x2 * input[1]);
       return sum;
}

double primeSigmoid(double net){
       return (net * (1 - net));
}

int main(){
    
    double lRate = 0.1;
    double x_1 = 0;
    double x_2 = 1;
    
    vector<double> target_output = {1, 0};     
    
    vector<double> inputWeight1 = {-1, 0};
    vector<double> inputWeight2 = {0, 1};
    
    vector<double> hiddenWeight1 = {1, 0};
    vector<double> hiddenWeight2 = {-1, 1};
    
    // first forward pass hidden layer output
    cout << "First Forward Pass"<< "\n";
    cout << "\n";
    for(;;){
    double hiddenNet1 = sumWeights(x_1, x_2, inputWeight1);
    double hiddenNet2 = sumWeights(x_1, x_2, inputWeight2);
    
    double hiddenNode1 = sigmoid(hiddenNet1);
    double hiddenNode2 = sigmoid(hiddenNet2);
    
    cout << "Hidden Node 1: " << hiddenNode1 << "\n";
    cout << "Hidden Node 2: " << hiddenNode2 << "\n";
    cout << "\n";
    
    double primeHidden1 =  primeSigmoid(hiddenNode1);
    double primeHidden2 =  primeSigmoid(hiddenNode2);
    
    // first forward pass output layer
    
    double outputNet1 = sumWeights(hiddenNode1, hiddenNode2, hiddenWeight1);
    double outputNet2 = sumWeights(hiddenNode1, hiddenNode2, hiddenWeight2);
    
    double outputNode1 = sigmoid(outputNet1);
    double outputNode2 = sigmoid(outputNet2);
    
    cout << "Output 1: " << outputNode1 << "\n";
    cout << "Output 2: " << outputNode2 << "\n";
    cout << "\n";
    
    if (outputNode1 == target_output[1] && outputNode2 == target_output[1]){
       break;
    }
    
    else {
       double primeOutput1 =  primeSigmoid(outputNode1);
       double primeOutput2 =  primeSigmoid(outputNode2);
       
       double outputError1 = (primeOutput1 * (target_output[0] - outputNode1));
       double outputError2 = (primeOutput2 * (target_output[1] - outputNode2));
       
       cout << "Output 1 Error: " << outputError1 << "\n";
       cout << "Output 2 Error: " << outputError2 << "\n";
       cout << "\n";
       
       // update output weights 
       double hiddenError1 = primeHidden1 * ((hiddenWeight1[0]*outputError1)+ (hiddenWeight2[0]*outputError2));
       double hiddenError2 = primeHidden2 * ((hiddenWeight1[1]*outputError1) + (hiddenWeight2[1]*outputError2));
       
       double dweight11 = (lRate * outputError1 * hiddenNode1);
       double dweight12 = (lRate * outputError1 * hiddenNode2);
       
       double dweight21 = (lRate * outputError2 * hiddenNode1);
       double dweight22 = (lRate * outputError2 * hiddenNode2);
       
       hiddenWeight1[0] = hiddenWeight1[0] + dweight11;
       hiddenWeight1[1] = hiddenWeight1[1] + dweight12;
       
       hiddenWeight2[0] = hiddenWeight2[0] + dweight21;
       hiddenWeight2[1] = hiddenWeight2[1] + dweight22;
       
       cout << "First Backward Pass" << "\n";
       cout << "\n";
       
       cout << "New Output Weights" << "\n";
       cout << "w11: " << hiddenWeight1[0] << " w12: " << hiddenWeight1[1] << " w21: " << hiddenWeight2[0] << " w22: " << hiddenWeight2[1] << "\n";
       cout << "\n";
       
       // update input layer weights
       
       
       cout << "Hidden Error 1: " << hiddenError1 << "\n";
       cout << "Hidden Error 2: " << hiddenError2 << "\n";
       cout << "\n";
       
       double dhweight11 = (lRate * hiddenError1 * x_1);
       double dhweight12 = (lRate * hiddenError2 * x_1);
       
       double dhweight21 = (lRate * hiddenError1 * x_2);
       double dhweight22 = (lRate * hiddenError2 * x_2);
       
       inputWeight1[0] = inputWeight1[0] + dhweight11;
       inputWeight1[1] = inputWeight1[1] + dhweight12;
       
       inputWeight2[0] = inputWeight2[0] + dhweight21;
       inputWeight2[1] = inputWeight2[1] + dhweight22;
       
       cout << "New Hidden Weights" << "\n";
       cout << "v11: " << inputWeight1[0] << " v12: " << inputWeight1[1] << " v21: " << inputWeight2[0] << " v22: " << inputWeight2[1] << "\n";
       cout << "\n";
    }
   }
}