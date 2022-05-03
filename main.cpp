#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX?1?7?1?7AVX2
#include<pthread.h>
#include<iostream>
#include<cmath>
#include <semaphore.h>
#include <stdio.h>
#include <windows.h>
#define ROW 1024
#define TASK 10
#define INTERVAL 10000
using namespace std;
float matrix[ROW][ROW];
typedef long long ll;
typedef struct {
	int k;
	int t_id;
}threadParam_t;

//?1?7?0?2?1?7?1?7?1?7?1?7?1?7?1?7��?1?7?1?7?1?7?1?0?1?7?0?0?1?7?1?9?1?7
sem_t sem_main;
sem_t sem_workerstart;
sem_t sem_workerend;
//信号量定义，用于静态线程
sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
//?1?7?0?0?0?5?1?7?1?7?1?7
pthread_barrier_t division;
pthread_barrier_t elemation;
//?1?7?1?7?0?0?1?7?1?9?1?7?1?7?1?7?1?7?1?7
int NUM_THREADS = 4;
//动态线程分配：待分配行数
int remain = ROW;
pthread_mutex_t remainLock;

void init()
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < i;j++)
			matrix[i][j] = 0;
		for(int j = i;j<ROW;j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0;k < 8000;k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0;j < ROW;j++)
			matrix[row1][j] += mult * matrix[row2][j];
	}
}

void plain() {
	for (int i = 0; i < ROW - 1; i++) {
		for (int j = i + 1; j < ROW; j++) {
			matrix[i][j] = matrix[i][j] / matrix[i][i];
		}
		matrix[i][i] = 1;
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
			}
			matrix[k][i] = 0;
		}
	}
}

void SIMD()
{
    for(int k = 0; k < ROW; ++k)
	{//?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		__m128 diver = _mm_load_ps1(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i += 1)
		{//?1?7?1?7?0?4
			__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
	}
}

void* dynamicFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int j = k + 1;j < ROW;++j)
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
	matrix[i][k] = 0;
	pthread_exit(NULL);
}

void* dynamicFuncSIMD(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
	int j;
	for (j = k + 1;j < ROW&& ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
	for (;j < ROW;j += 4)
	{
		__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
		__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
		mult2 = _mm_mul_ps(mult1, mult2);
		sub1 = _mm_sub_ps(sub1, mult2);
		_mm_storeu_ps(&matrix[i][j], sub1);
	}
	matrix[i][k] = 0;
	pthread_exit(NULL);
}

void dynamicMain(void* (*threadFunc)(void*))
{
	for(int k = 0; k < ROW; ++k)
	{//?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		__m128 diver = _mm_load_ps1(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?0?1?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?1?7?1?7?1?7?1?7
		int worker_count = ROW-1-k; //?1?7?1?7?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?7
		pthread_t* handles = new pthread_t[worker_count];// ?1?7?1?7?1?7?1?7?1?7?1?7?0?8?1?7?1?7 Handle
		threadParam_t* param = new threadParam_t[worker_count];// ?1?7?1?7?1?7?1?7?1?7?1?7?0?8?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?1?5?5
		//?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//?1?7?1?7?1?7?1?7?1?7?1?9?1?7
		for(int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//?1?7?1?7?1?7?1?9?0?3?1?7?1?7?1?7?0?9?1?7?1?7?1?7?1?7��?0?1?1?7?1?7?1?7?1?7?1?9?1?7?1?7?1?7?0?1?1?7?1?7?1?7?1?7?1?7?0?4?1?7?1?7?1?7?1?7
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* newDynamicFuncSIMD(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int i = k + t_id + 1;i < ROW;i += NUM_THREADS)
	{
		__m128 mult1 = _mm_load_ss(&matrix[i][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
		for (;j < ROW;j += 4)
		{
			__m128 sub1 = _mm_loadu_ps(&matrix[i][j]);
			__m128 mult2 = _mm_loadu_ps(&matrix[k][j]);
			mult2 = _mm_mul_ps(mult1, mult2);
			sub1 = _mm_sub_ps(sub1, mult2);
			_mm_storeu_ps(&matrix[i][j], sub1);
		}
		matrix[i][k] = 0;
	}
	pthread_exit(NULL);
}

void newDynamicMain(void* (*threadFunc)(void*))
{
	for (int k = 0; k < ROW; ++k)
	{//主线程做除法操作
		__m128 diver = _mm_load_ss(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = NUM_THREADS; //工作线程数量
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* staticFunc(void* param) {
	long t_id = (long long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//sem_wait(&sem_workerstart); // ?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?9?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?0?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		pthread_barrier_wait(&division);
		//?0?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//?1?7?1?7?0?4
			__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		pthread_barrier_wait(&elemation);
        //sem_post(&sem_main); // ?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?1?7
    	//sem_wait(&sem_workerend); //?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?9?1?7?1?7?1?7?1?7?1?9?0?5?1?7?1?7?0?3?1?7?1?7?1?7?1?7?1?7?0?5?1?7?1?7
	}
    pthread_exit(NULL);
}

void staticMain()
{
	//初始化barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS+1);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS+1);
	pthread_t* handles = new pthread_t[NUM_THREADS];// ?1?7?1?7?1?7?1?7?1?7?1?7?0?8?1?7?1?7 Handle
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, staticFunc, (void*)t_id);

	for(int k = 0; k < ROW; ++k)
	{
		//?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		__m128 diver = _mm_load_ps1(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		pthread_barrier_wait(&division);
		pthread_barrier_wait(&elemation);
	}
    for(int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void* staticFuncOpt(void* param) {
	long t_id = (long long)param;
	for (int k = 0; k < ROW; ++k)
	{
		// t_id ?0?2 0 ?1?7?1?7?1?7?1?9?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?1?7?1?7?0?0?0?9?1?7
		// ?1?7?1?7?1?7?1?7?0?1?1?7?1?7?1?7?1?7?1?7?1?7?0?5?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?0?2?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?0?1?1?7?0?6?1?7?1?7?0?5?1?7?1?7?0?4?1?7?1?7?0?2?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?1?7?1?7?1?7?0?0?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		// ?1?7?1?7?1?7?0?2?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?1?7?1?7?1?7?1?7?0?4?1?7?1?7?0?0?1?7?1?7 barrier
		if (t_id == 0)
		{
			__m128 diver = _mm_set1_ps(matrix[k][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
				matrix[k][j] = matrix[k][j] / matrix[k][k];
			for (;j < ROW;j += 4)
			{
				__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&matrix[k][j], divee);
			}
			matrix[k][k] = 1.0;
		}
		else sem_wait(&sem_Divsion[t_id-1]); // ?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?9?1?7?1?7?1?7?0?0?1?7?1?7?1?7?1?7?1?7?1?7?1?7
		// t_id ?0?2 0 ?1?7?1?7?1?7?1?9?0?5?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?9?0?1?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?1?7?1?7?1?7?1?7
		if (t_id == 0)
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Divsion[t_id]);

		//?0?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?0?1?1?7?0?6?1?7?1?7?0?5?1?7?1?7?0?8?1?7?1?7?1?7?1?7?1?7?1?7?0?1?0?7?1?7?0?4?1?7?1?7
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//?1?7?1?7?0?4
			__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// ?1?7?1?7?1?7?1?7?1?7?1?9?1?7?0?5?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?5?1?7?1?7
		if (t_id == 0)
		{
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_wait(&sem_leader);
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Elimination[t_id]);
		}
		else
		{
			sem_post(&sem_leader);
			sem_wait(&sem_Elimination[t_id-1]);
		}
	}
	pthread_exit(NULL);
}

void staticOptMain(void* (*threadFunc)(void*))
{
	//?1?7?1?7?0?3?1?7?1?7?1?7?0?2?1?7?1?7?1?7
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < NUM_THREADS-1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	//?1?7?1?7?1?7?1?7?1?7?1?9?1?7
	pthread_t* handles = new pthread_t[NUM_THREADS];// ?1?7?1?7?1?7?1?7?1?7?1?7?0?8?1?7?1?7 Handle
	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for(int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(&sem_leader);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{ 
		sem_destroy(&sem_Divsion[t_id]);
		sem_destroy(&sem_Elimination[t_id]);
	}
}

void* staticFuncOptNew(void* param) {
	long t_id = (long long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//?1?7?1?7?1?7?1?7
		int count = (ROW - k - 1) / NUM_THREADS;
		__m128 diver = _mm_load_ps1(&matrix[k][k]);
		int j;
		//?1?7?1?7?1?7?1?9?0?8?1?7?1?7?1?7k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//?1?7?1?7?1?7?1?7?0?6?1?7?1?7
		for (j = k + 1 + count*t_id;j < endIt && ((endIt - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < endIt;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//?0?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?4?0?1?1?7?0?6?1?7?1?7?0?5?1?7?1?7?0?8?1?7?1?7?1?7?1?7?1?7?1?7?0?1?0?7?1?7?0?4?1?7?1?7
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{//?1?7?1?7?0?4
			__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//?1?7?1?7?1?7��?1?7?1?7?1?7?1?7?1?7?1?7?1?7
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// ?1?7?1?7?1?7?1?7?1?7?1?9?1?7?0?5?1?7?1?7?1?7?1?7?1?7?1?7?1?7?0?5?1?7?1?7
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticNewOptMain(void* (*threadFunc)(void*))
{
	//初始化barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		__m128 diver = _mm_load_ss(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW - count;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee = _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		pthread_barrier_wait(&division);
		for (int i = k + NUM_THREADS; i < ROW; i += NUM_THREADS)
		{//消去
			__m128 mult1 = _mm_load_ss(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 = _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 = _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void DynamicDivMain(void* (*threadFunc)(void*))
{
	//初始化锁
	pthread_mutex_init(&remainLock, NULL);
	//初始化barrier
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
	long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	//主函数看作第NUM_THREADS-1号线程
	for (int k = 0;k < ROW;++k)
	{
		__m128 diver = _mm_load_ss(&matrix[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
		//主线程处理ROW-count~ROW-1
		for (j = ROW - count;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee = _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		//发出任务
		remain = k + 1;
		//等待子线程就位
		pthread_barrier_wait(&division);
		while(true)
		{
			int i;//行
			//处理i`i+TASK - 1
			//领取任务
			pthread_mutex_lock(&remainLock);
			if (remain >= ROW) 
			{
				pthread_mutex_unlock(&remainLock);
				break;
			}
			i = remain;
			remain += TASK;
			pthread_mutex_unlock(&remainLock);
			int end = min(ROW, i + TASK);
			for (;i < end;i++)
			{
				//消去
				__m128 mult1 = _mm_load_ss(&matrix[i][k]);
				int j;//列
				for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (;j < ROW;j += 4)
				{
					__m128 sub1 = _mm_loadu_ps(&matrix[i][j]);
					__m128 mult2 = _mm_loadu_ps(&matrix[k][j]);
					mult2 = _mm_mul_ps(mult1, mult2);
					sub1 = _mm_sub_ps(sub1, mult2);
					_mm_storeu_ps(&matrix[i][j], sub1);
				}
				matrix[i][k] = 0.0;
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
	pthread_mutex_destroy(&remainLock);
}

void* DynamicDivFunc(void* param) {
	long t_id = (long long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//除法
		int count = (ROW - k - 1) / NUM_THREADS;
		__m128 diver = _mm_load_ss(&matrix[k][k]);
		int j;
		//子线程处理k+1+count*t_id~k+count*(t_id+1)
		int endIt = k + 1 + count * (t_id + 1);//向量末端
		for (j = k + 1 + count * t_id;j < endIt && ((endIt - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < endIt;j += 4)
		{
			__m128 divee = _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		pthread_barrier_wait(&division);
		//循环划分任务（同学们可以尝试多种任务划分方式）
		while (true)
		{
			int i;
			pthread_mutex_lock(&remainLock);
			if (remain >= ROW)
			{
				pthread_mutex_unlock(&remainLock);
				break;
			}
			i = remain;
			remain += TASK;
			pthread_mutex_unlock(&remainLock);
			int end = min(ROW, i + TASK);
			for (; i < end; i++)
			{//消去
				__m128 mult1 = _mm_load_ss(&matrix[i][k]);
				int j;
				for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
					matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
				for (;j < ROW;j += 4)
				{
					__m128 sub1 = _mm_loadu_ps(&matrix[i][j]);
					__m128 mult2 = _mm_loadu_ps(&matrix[k][j]);
					mult2 = _mm_mul_ps(mult1, mult2);
					sub1 = _mm_sub_ps(sub1, mult2);
					_mm_storeu_ps(&matrix[i][j], sub1);
				}
				matrix[i][k] = 0.0;
			}
		}
		// 所有线程一起进入下一轮
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void timing(void (*func)())
{
	ll head, tail, freq;
	double time = 0;
	int counter = 0;
	while (INTERVAL > time)
	{
        init();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
		func();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		counter++;
		time += (tail - head) * 1000.0 / freq;
	}
	std::cout << time / counter << '\n';
}

void timing(void (*func)(void* (*threadFunc)(void*)), void* (*threadFunc)(void*))
{
	ll head, tail, freq;
	double time = 0;
	int counter = 0;
	while (INTERVAL > time)
	{
        init();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
		func(threadFunc);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		counter++;
		time += (tail - head) * 1000.0 / freq;
	}
	std::cout << time / counter << '\n';
}

int main()
{
	cout<<"plain: ";
	timing(plain);
	cout<<"SIMD: ";
	timing(SIMD);
	for(NUM_THREADS = 8;NUM_THREADS<=8;NUM_THREADS++)
	{
		cout<<"using "<<NUM_THREADS<<" threads"<<endl;
		cout << "dynamic(reasonable): ";
		timing(newDynamicMain, newDynamicFuncSIMD);
		cout<<"static: ";
		timing(staticMain);
		cout<<"static(add division): ";
		timing(staticOptMain, staticFuncOpt);
		cout << "static(myOpt row div): ";
		timing(staticNewOptMain, staticFuncOptNew);
		cout << "dynamicDiv: ";
		timing(DynamicDivMain, DynamicDivFunc);
	}
		cout<<"dynamic: ";
		timing(dynamicMain, dynamicFunc);
		cout<<"dynamic+SIMD: ";
		timing(dynamicMain, dynamicFuncSIMD);
}
