# hpc

##### Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS.

+ Program
   ```cpp
   #include <iostream>
   #include <queue>
   #include <vector>
   #include <omp.h>

   using namespace std;

   // Graph class
   class Graph {
   private:
       int V;
       vector<int>* adj;

   public:
       Graph(int V) {
           this->V = V;
           adj = new vector<int>[V];
       }

       void addEdge(int v, int w) {
           adj[v].push_back(w);
           adj[w].push_back(v);
       }

       void bfs(int start) {
           vector<bool> visited(V, false);
           queue<int> q;

           visited[start] = true;
           q.push(start);

           while (!q.empty()) {
               int u = q.front();
               q.pop();
               cout << u << " ";

               #pragma omp parallel for
               for (int i = 0; i < adj[u].size(); i++) {
                   int v = adj[u][i];
                   if (!visited[v]) {
                       visited[v] = true;
                       q.push(v);
                   }
               }
           }
       }

       void dfs(int start) {
           vector<bool> visited(V, false);

           dfs_helper(start, visited);
       }

   private:
       void dfs_helper(int u, vector<bool>& visited) {
           visited[u] = true;
           cout << u << " ";

           #pragma omp parallel for
           for (int i = 0; i < adj[u].size(); i++) {
               int v = adj[u][i];
               if (!visited[v]) {
                   dfs_helper(v, visited);
               }
           }
       }
   };

   int main() {
       Graph g(6);

       g.addEdge(0, 1);
       g.addEdge(0, 2);
       g.addEdge(1, 3);
       g.addEdge(2, 4);
       g.addEdge(3, 4);
       g.addEdge(3, 5);

       cout << "BFS starting from vertex 3: ";
       g.bfs(3);
       cout << endl;

       cout << "DFS starting from vertex 5: ";
       g.dfs(5);
       cout << endl;

       return 0;
   }
   ```
+ Output
   ```bash
   BFS starting from vertex 3: 3 1 4 5 0 2 
   DFS starting from vertex 5: 5 3 1 0 2 4 
   ```
***
##### Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms.
+ Program
   ```cpp
   #include <iostream>
   #include <vector>
   #include <algorithm>
   #include <omp.h>
   using namespace std;

   // Parallel bubble sort implementation
   void parallel_bubble_sort(vector<int>& arr) {
       int n = arr.size();
       bool swapped = true;
       while (swapped) {
           swapped = false;
           #pragma omp parallel for shared(arr)
           for (int i = 1; i < n; ++i) {
               if (arr[i - 1] > arr[i]) {
                   swap(arr[i - 1], arr[i]);
                   swapped = true;
               }
           }
       }
   }

   // Parallel merge sort implementation
   void parallel_merge_sort(vector<int>& arr) {
       if (arr.size() > 1) {
           vector<int> left(arr.begin(), arr.begin() + arr.size() / 2);
           vector<int> right(arr.begin() + arr.size() / 2, arr.end());
           #pragma omp parallel sections
           {
               #pragma omp section
               parallel_merge_sort(left);
               #pragma omp section
               parallel_merge_sort(right);
           }
           merge(left.begin(), left.end(), right.begin(), right.end(), arr.begin());
       }
   }

   void show(int op, vector<int>& arr){
       vector<int> copy = arr;    string str="", name="";
       switch(op){
           case 0:    name="Original";   str=" without";     break;
           case 1:    name="Sequential"; str="bubble";   
                      sort(copy.begin(), copy.end());        break;
           case 2:    name="Parallel";   str="  bubble";
                      parallel_bubble_sort(copy);            break;
           case 3:    name="Sequential"; str=" merge";
                      stable_sort(copy.begin(), copy.end()); break;
           case 4:    name="Parallel";   str=" merge";
                      parallel_merge_sort(copy);             break;
       }
       cout << name  << " " << str << " sort : ";
       for (const auto& num : copy) cout << num << " ";
       cout << endl;
   }

   int main() {
       vector<int> arr{ 4, 2, 6, 8, 1, 3, 9, 5, 7 };
       for(int i=0; i<5; i++) show(i, arr);
       return 0;
   }
   ```
+ Output
   ```bash
   Original  without sort : 4 2 6 8 1 3 9 5 7 
   Sequential bubble sort : 1 2 3 4 5 6 7 8 9 
   Parallel   bubble sort : 1 2 3 4 5 6 7 8 9 
   Sequential  merge sort : 1 2 3 4 5 6 7 8 9 
   Parallel    merge sort : 1 2 3 4 5 6 7 8 9 
   ```
***
##### Implement Min, Max, Sum and Average operations using Parallel Reduction.
+ Program
   ```cpp
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
   ```
+ Output
   ```bash
   Data :
    86 77 15 93 35 86 92 49 21 62
    27 90 59 63 26 40 26 72 36 11
    68 67 29 82 30 62 23 67 35 29
    2 22 58 69 67 93 56 11 42 29
    73 21 19 84 37 98 24 15 70 13
    26 91 80 56 73 62 70 96 81 5
    25 84 27 36 5 46 29 13 57 24
    95 82 45 14 67 34 64 43 50 87
    8 76 78 88 84 3 51 54 99 32
    60 76 68 39 12 26 86 94 39 0

   Minimum : 2
   Maximum : 99
   Sum     : 5184
   Average : 51.84
   ```
