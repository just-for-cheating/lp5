#include <iostream>
#include <omp.h>
#include <time.h>
#include <iomanip>
using namespace std;

void merge(int N, int *arr, int p, int q, int r)
{
    int n1 = q - p + 1;
    int n2 = r - q;

    int *arr1 = new int[n1];
    int *arr2 = new int[n2];

    for (int i = 0; i < n1; i++)
    {
        arr1[i] = arr[p + i];
    }

    for (int j = 0; j < n2; j++)
    {
        arr2[j] = arr[q + j + 1];
    }

    int i, j, k;
    i = 0;
    j = 0;
    k = p;

    while (i < n1 && j < n2)
    {
        if (arr1[i] <= arr2[j])
        {
            arr[k] = arr1[i];
            i++;
        }
        else
        {
            arr[k] = arr2[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = arr1[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = arr2[j];
        j++;
        k++;
    }
}

void mergeSort(int N, int *arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        mergeSort(N, arr, l, m);
        mergeSort(N, arr, m + 1, r);

        merge(N, arr, l, m, r);
    }
}

void mergeSortParallel(int N, int *arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

#pragma omp parallel sections num_threads(2)
        {

#pragma omp section
            {
                mergeSort(N, arr, l, m);
            }

#pragma omp section
            {
                mergeSort(N, arr, m + 1, r);
            }
        }
        merge(N, arr, l, m, r);
    }
}

void bubbleSort(int N, int *arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(int N, int *arr)
{
    int f;
    for (int i = 0; i < N; i++)
    {
        f = i % 2;

#pragma omp parallel for default(none), shared(arr, f, N)
        for (int j = f; j < N - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main()
{
    clock_t start, end;
    int N;
    cout << "Size of Array : ";
    cin >> N;

    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];
    int *d = new int[N];
    int *e = new int[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 1000;
        b[i] = a[i];
        c[i] = a[i];
        d[i] = a[i];
        e[i] = a[i];
    }

    // Bubble Sort
    start = clock();
    bubbleSort(N, a);
    end = clock();
    float sBubbleTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "\nBubble Sort Time : " << fixed << setprecision(9) << sBubbleTime << endl;

    // Parallel Bubble Sort
    start = clock();
    parallelBubbleSort(N, b);
    end = clock();
    float pBubbleTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "Parallel Bubble Sort : " << fixed << setprecision(9) << pBubbleTime << endl;

    // Compare Bubble Sort
    cout << "Speed Up of Bubble Sort : " << fixed << setprecision(9) << sBubbleTime / pBubbleTime << endl;

    // Merge Sort
    start = clock();
    mergeSort(N, c, 0, N - 1);
    end = clock();
    float sMergeTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "\nSerial Merge Sort : " << fixed << setprecision(9) << sMergeTime << endl;

    // Parallel Merge Sort
    start = clock();
    mergeSortParallel(N, d, 0, N - 1);
    end = clock();
    float pMergeTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "Parallel Merge Sort : " << fixed << setprecision(9) << pMergeTime << endl;

    // Compare Merge Sort
    cout << "Speed Up of Merge Sort Section : " << fixed << setprecision(9) << sMergeTime / pMergeTime << endl;
}