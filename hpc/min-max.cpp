#include <iostream>
#include <omp.h>
#include <time.h>
#include <chrono>
#include <limits>
using namespace std;
using namespace std::chrono;

using namespace std;

int sum(int a[], int n)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
}

int min(int a[], int n)
{
    int v = a[0];
#pragma omp parallel for reduction(min : v)
    for (int i = 0; i < n; i++)
    {
        if (a[i] < v)
            v = a[i];
    }
    return v;
}

int max(int a[], int n)
{
    int v = a[0];
#pragma omp parallel for reduction(max : v)
    for (int i = 0; i < n; i++)
    {
        if (a[i] > v)
            v = a[i];
    }
    return v;
}

float avg(int a[], int n)
{
    return sum(a, n) / n;
}

int main()
{
    int n = 1000;
    int a[n];
    int cnt = 0;

    for (int i = 0; i < n; i++)
    {
        a[i] = i + 5;
        cnt = cnt + 1;
    }

    cout << " Input data is :";
    for (int i = 0; i < cnt; i++)
    {
        cout << " " << a[i];
    }

    cout << "\n";

    // Time taken for sum
    auto beg = high_resolution_clock::now();
    int sumVal = sum(a, n);
    auto end = high_resolution_clock::now();
    auto sumDuration = duration_cast<microseconds>(end - beg);

    // Time taken for average
    beg = high_resolution_clock::now();
    int avgVal = avg(a, n);
    end = high_resolution_clock::now();
    auto avgDuration = duration_cast<microseconds>(end - beg);

    // Time taken for min
    beg = high_resolution_clock::now();
    int minVal = min(a, n);
    end = high_resolution_clock::now();
    auto minDuration = duration_cast<microseconds>(end - beg);

    // Time taken by max
    beg = high_resolution_clock::now();
    int maxVal = max(a, n);
    end = high_resolution_clock::now();
    auto maxDuration = duration_cast<microseconds>(end - beg);

    cout << "Sum of given numbers :  " << sumVal << endl;
    cout << "Time for sum is :  " << sumDuration.count() << endl;
    cout << "Min from numbers is :  " << minVal << endl;
    cout << "Time for min is : " << minDuration.count() << endl;
    cout << "max from numbers is :  " << maxVal << endl;
    cout << "Time for max is : " << maxDuration.count() << endl;
    cout << "Average of given numbers is : " << avgVal << endl;
    cout << "Time for average : " << avgDuration.count() << endl;
    cout << "\n";

    return 0;
}