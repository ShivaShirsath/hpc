#include <iostream>
#include <vector>
   #include <algorithm>
   #include <numeric>
   #include <limits>
   #include <omp.h>

   using namespace std;

   int findMin(vector<int>& data) {
       int minVal = numeric_limits<int>::max();
       #pragma omp parallel for reduction(min:minVal)
       for (int i = 0; i < data.size(); i++) if (data[i] < minVal) minVal = data[i];
       return minVal;
   }

   int findMax(vector<int>& data) {
       int maxVal = numeric_limits<int>::min();
       #pragma omp parallel for reduction(max:maxVal)
       for (int i = 0; i < data.size(); i++) if (data[i] > maxVal) maxVal = data[i];
       return maxVal;
   }

   int findSum(vector<int>& data) {
       int sum = 0;
       #pragma omp parallel for reduction(+:sum)
       for (int i = 0; i < data.size(); i++) sum += data[i];
       return sum;
   }

   double findAverage(vector<int>& data) {
       double sum = 0;
       #pragma omp parallel for reduction(+:sum)
       for (int i = 0; i < data.size(); i++) sum += data[i];
       return sum / data.size();
   }

   int main() {
       vector<int> data(100);
       generate(data.begin(), data.end(), [](){ return rand() % 100; });
       cout << "Data :" << endl;
       for (int i = 1; i < data.size()+1; i++) {
           cout << " " << data[i] ;    if(i%10==0) cout << endl;
       }    cout << endl;
       cout << "Minimum : " <<     findMin(data) << endl;
       cout << "Maximum : " <<     findMax(data) << endl;
       cout << "Sum     : " <<     findSum(data) << endl;
       cout << "Average : " << findAverage(data) << endl;
       return 0;
   }